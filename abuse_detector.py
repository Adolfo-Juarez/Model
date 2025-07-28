import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import joblib
import json

load_dotenv()

# Configuración de la conexión a la base de datos
def get_database_connection(host=os.environ.get("MYSQL_HOST"), port=os.environ.get("MYSQL_PORT"), 
                           database=os.environ.get("MYSQL_DATABASE"), 
                           user=os.environ.get("MYSQL_USERNAME"), password=os.environ.get("MYSQL_PASSWORD")):
    """
    Crea una conexión a la base de datos MySQL.
    """
    connection_string = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def load_data_from_db(engine, patient_id=None):
    """
    Carga los datos desde la base de datos.
    
    Args:
        engine: Conexión a la base de datos
        patient_id: Si se especifica, filtra solo ese paciente. Si es None, obtiene todos.
    """
    base_query = """
    SELECT 
        r.id as id_receta,
        rm.medication_id as id_medicamento,
        r.patient_id as id_paciente,
        r.doctor_id as id_medico,
        CASE
            WHEN rm.supplied = true THEN 'success'
            ELSE 'failed'
        END as status,
        COALESCE(rm.supplied_at, i.created_at) as date
    FROM
        recipe_medication rm
    INNER JOIN recipe r ON
        rm.recipe_id = r.id
    LEFT JOIN incidence i ON
        i.created_at IS NOT NULL
    WHERE
        rm.supplied IS NOT NULL
        OR i.id IS NOT NULL
    """
    
    if patient_id:
        query = text(base_query + f"HAVING r.patient_id = '{patient_id}' ORDER BY date ASC")
    else:
        query = text(base_query + "ORDER BY date ASC")
    
    try:
        return pd.read_sql_query(query, engine)
    except Exception as e:
        print(f"Error al ejecutar la consulta: {e}")
        print("Intentando cargar desde archivo CSV...")
        return pd.read_csv("recetas.csv")

def extract_abuse_features(df):
    """
    Extrae features específicas para detectar comportamiento abusivo de usuarios.
    """
    # Convertir fecha
    df['date'] = pd.to_datetime(df['date'])
    
    # Features base por paciente
    df_features = df.groupby('id_paciente').agg(
        total_recetas=('id_receta', 'count'),
        recetas_exitosas=('status', lambda x: (x == 'success').sum()),
        recetas_fallidas=('status', lambda x: (x == 'failed').sum()),
        doctores_distintos=('id_medico', 'nunique'),
        medicamentos_distintos=('id_medicamento', 'nunique'),
        primera_receta=('date', 'min'),
        ultima_receta=('date', 'max'),
    ).reset_index()
    
    # Features derivadas para detectar abuso
    df_features['tasa_exito'] = df_features['recetas_exitosas'] / df_features['total_recetas']
    df_features['duracion_actividad_dias'] = (df_features['ultima_receta'] - df_features['primera_receta']).dt.days + 1
    df_features['frecuencia_recetas'] = df_features['total_recetas'] / df_features['duracion_actividad_dias']
    df_features['frecuencia_exitosas'] = df_features['recetas_exitosas'] / df_features['duracion_actividad_dias']
    
    # Indicadores de doctor shopping
    df_features['ratio_doctores_recetas'] = df_features['doctores_distintos'] / df_features['total_recetas']
    df_features['es_doctor_shopper'] = (df_features['doctores_distintos'] >= 3).astype(int)
    
    # Indicadores de concentración en medicamentos
    df_features['ratio_medicamentos_recetas'] = df_features['medicamentos_distintos'] / df_features['total_recetas']
    df_features['concentracion_medicamentos'] = 1 / df_features['medicamentos_distintos']
    
    # Indicadores temporales
    df_features['actividad_reciente'] = (
        (datetime.now() - df_features['ultima_receta']).dt.days <= 30
    ).astype(int)
    
    # Análisis de patrones por medicamento específico
    medicamento_stats = df[df['status'] == 'success'].groupby(['id_paciente', 'id_medicamento']).size().reset_index(name='count')
    if len(medicamento_stats) > 0:
        medicamento_max = medicamento_stats.groupby('id_paciente')['count'].max().reset_index()
        medicamento_max.columns = ['id_paciente', 'max_medicamento_repetido']
        df_features = df_features.merge(medicamento_max, on='id_paciente', how='left')
    else:
        df_features['max_medicamento_repetido'] = 0
    
    df_features['max_medicamento_repetido'] = df_features['max_medicamento_repetido'].fillna(0)
    
    # Score de comportamiento temporal sospechoso
    def calcular_patron_temporal(group):
        dates = pd.to_datetime(group['date']).sort_values()
        if len(dates) < 2:
            return 0
        
        diff_days = dates.diff().dt.days.dropna()
        recetas_rapidas = (diff_days <= 7).sum()
        patron_intenso = recetas_rapidas / len(diff_days) if len(diff_days) > 0 else 0
        
        return patron_intenso
    
    # Patrón temporal
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        patron_temporal = df[df['status'] == 'success'].groupby('id_paciente').apply(calcular_patron_temporal)
    
    if len(patron_temporal) > 0:
        patron_temporal = patron_temporal.reset_index()
        patron_temporal.columns = ['id_paciente', 'patron_temporal_sospechoso']
        df_features = df_features.merge(patron_temporal, on='id_paciente', how='left')
    else:
        df_features['patron_temporal_sospechoso'] = 0
    
    df_features['patron_temporal_sospechoso'] = df_features['patron_temporal_sospechoso'].fillna(0)
    
    return df_features

def build_and_train_model(contamination=0.1, model_path="drug_abuse_model.pkl"):
    """
    Construye, entrena y guarda el modelo de detección de abuso.
    
    Args:
        contamination: Porcentaje esperado de casos anómalos
        model_path: Ruta donde guardar el modelo
    
    Returns:
        dict: Información sobre el entrenamiento
    """
    print("Iniciando construcción y entrenamiento del modelo...")
    
    # Cargar datos de entrenamiento (todos los pacientes)
    engine = get_database_connection()
    if not engine:
        raise Exception("No se pudo conectar a la base de datos para entrenar el modelo")
    
    df = load_data_from_db(engine)
    print(f"Datos de entrenamiento cargados: {len(df)} registros")
    
    # Extraer features
    df_features = extract_abuse_features(df)
    print(f"Features extraídas para {len(df_features)} pacientes")
    
    if len(df_features) < 10:
        raise Exception("Datos insuficientes para entrenar el modelo (mínimo 10 pacientes)")
    
    # Features para el modelo
    feature_columns = [
        'frecuencia_exitosas',
        'tasa_exito',
        'doctores_distintos',
        'ratio_doctores_recetas',
        'concentracion_medicamentos',
        'max_medicamento_repetido',
        'patron_temporal_sospechoso',
    ]
    
    X = df_features[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelo
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_scaled)
    
    # Guardar modelo, scaler y columnas
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_stats': {
            'total_patients': len(df_features),
            'total_records': len(df),
            'contamination': contamination,
            'training_date': datetime.now().isoformat()
        }
    }
    
    joblib.dump(model_data, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    return model_data['training_stats']

def load_trained_model(model_path="drug_abuse_model.pkl"):
    """
    Carga un modelo previamente entrenado.
    
    Args:
        model_path: Ruta del modelo guardado
    
    Returns:
        tuple: (model, scaler, feature_columns)
    """
    try:
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['scaler'], model_data['feature_columns']
    except FileNotFoundError:
        raise Exception(f"Modelo no encontrado en {model_path}. Entrena el modelo primero.")

def detect_patient_abuse(patient_id, model_path="drug_abuse_model.pkl"):
    """
    Detecta comportamiento abusivo para un paciente específico usando modelo entrenado.
    
    Args:
        patient_id: ID del paciente a analizar
        model_path: Ruta del modelo entrenado
    
    Returns:
        dict: Resultado de la detección en formato JSON
    """
    try:
        # Cargar modelo entrenado
        model, scaler, feature_columns = load_trained_model(model_path)
        
        # Cargar datos del paciente específico
        engine = get_database_connection()
        if not engine:
            raise Exception("No se pudo conectar a la base de datos")
        
        df = load_data_from_db(engine, patient_id=patient_id)
        
        if len(df) == 0:
            return {
                "patient_id": patient_id,
                "status": "error",
                "message": "No se encontraron registros para este paciente",
                "abuse_detected": False,
                "confidence_score": 0.0,
                "details": {}
            }
        
        # Extraer features
        df_features = extract_abuse_features(df)
        
        if len(df_features) == 0:
            return {
                "patient_id": patient_id,
                "status": "error",
                "message": "No se pudieron extraer features para este paciente",
                "abuse_detected": False,
                "confidence_score": 0.0,
                "details": {}
            }
        
        patient_data = df_features.iloc[0]  # Solo hay un paciente
        
        # Preparar datos para predicción
        X = df_features[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predicción
        prediction = model.predict(X_scaled)[0]
        anomaly_score = model.decision_function(X_scaled)[0]
        
        # Convertir score a porcentaje de sospecha (0-100%)
        # Score más negativo = más sospechoso
        confidence_score = max(0, min(100, (abs(anomaly_score) * 100)))
        
        abuse_detected = prediction == -1
        
        # Crear respuesta JSON
        result = {
            "patient_id": patient_id,
            "status": "success",
            "abuse_detected": bool(abuse_detected),
            "confidence_score": round(confidence_score, 2),
            "anomaly_score": round(float(anomaly_score), 4),
            "analysis_date": datetime.now().isoformat(),
            "details": {
                "total_prescriptions": int(patient_data['total_recetas']),
                "successful_prescriptions": int(patient_data['recetas_exitosas']),
                "failed_prescriptions": int(patient_data['recetas_fallidas']),
                "success_rate": round(float(patient_data['tasa_exito']), 4),
                "unique_doctors": int(patient_data['doctores_distintos']),
                "unique_medications": int(patient_data['medicamentos_distintos']),
                "daily_frequency": round(float(patient_data['frecuencia_exitosas']), 4),
                "doctor_shopping_indicator": bool(patient_data['es_doctor_shopper']),
                "max_repeated_medication": int(patient_data['max_medicamento_repetido']),
                "temporal_pattern_suspicious": round(float(patient_data['patron_temporal_sospechoso']), 4),
                "activity_duration_days": int(patient_data['duracion_actividad_dias']),
                "recent_activity": bool(patient_data['actividad_reciente'])
            },
            "risk_factors": []
        }
        
        # Identificar factores de riesgo específicos
        if patient_data['doctores_distintos'] >= 3:
            result["risk_factors"].append("Demasiadas recetas emitidas por una variedad extensa de doctores")
        
        if patient_data['frecuencia_exitosas'] < 0.5:
            result["risk_factors"].append("Baja frecuencia de prescripciones exitosas")
        
        if patient_data['tasa_exito'] < 0.3:
            result["risk_factors"].append("Tasa de éxito sospechosamente baja")
        
        if patient_data['max_medicamento_repetido'] > 2:
            result["risk_factors"].append("Repetición excesiva de medicamento específico")
        
        if patient_data['patron_temporal_sospechoso'] < 0.3:
            result["risk_factors"].append("Patrón temporal sospechoso (recetas muy seguidas)")
        
        return result
        
    except Exception as e:
        return {
            "patient_id": patient_id,
            "status": "error",
            "message": str(e),
            "abuse_detected": False,
            "confidence_score": 0.0,
            "details": {}
        }

def generate_batch_report(model_path="drug_abuse_model.pkl", contamination_threshold=0.1):
    """
    Genera un reporte de todos los pacientes usando el modelo entrenado.
    
    Returns:
        dict: Reporte completo en formato JSON
    """
    try:
        # Cargar modelo
        model, scaler, feature_columns = load_trained_model(model_path)
        
        # Cargar todos los datos
        engine = get_database_connection()
        df = load_data_from_db(engine)
        df_features = extract_abuse_features(df)
        
        # Preparar datos
        X = df_features[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predicciones
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        df_features['abuse_detected'] = (predictions == -1)
        df_features['anomaly_score'] = anomaly_scores
        
        # Estadísticas generales
        total_patients = len(df_features)
        suspicious_patients = df_features['abuse_detected'].sum()
        
        # Pacientes más sospechosos
        top_suspicious = df_features[df_features['abuse_detected'] == True].nsmallest(10, 'anomaly_score')
        
        suspicious_list = []
        for _, patient in top_suspicious.iterrows():
            suspicious_list.append({
                "patient_id": patient['id_paciente'],
                "anomaly_score": round(float(patient['anomaly_score']), 4),
                "total_prescriptions": int(patient['total_recetas']),
                "success_rate": round(float(patient['tasa_exito']), 4),
                "unique_doctors": int(patient['doctores_distintos']),
                "daily_frequency": round(float(patient['frecuencia_exitosas']), 4)
            })
        
        return {
            "status": "success",
            "analysis_date": datetime.now().isoformat(),
            "summary": {
                "total_patients_analyzed": total_patients,
                "suspicious_patients": int(suspicious_patients),
                "suspicious_percentage": round((suspicious_patients / total_patients) * 100, 2)
            },
            "top_suspicious_patients": suspicious_list
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Funciones principales de uso
def train_model():
    """Entrena y guarda el modelo"""
    return build_and_train_model()

def analyze_patient(patient_id):
    """Analiza un paciente específico"""
    return detect_patient_abuse(patient_id)

def analyze_all_patients():
    """Analiza todos los pacientes"""
    return generate_batch_report()

# Ejemplo de uso
if __name__ == "__main__":
    # Opción 1: Entrenar modelo
    # print("Entrenando modelo...")
    # train_stats = train_model()
    # print(json.dumps(train_stats, indent=2))
    
    # Opción 2: Analizar paciente específico
    # patient_id = "e30afed4-8fb3-4fec-b140-9626b9bc94ea"
    # print(f"Analizando paciente: {patient_id}")
    # result = analyze_patient(patient_id)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Opción 3: Analizar todos los pacientes
    # print("Generando reporte completo...")
    # report = analyze_all_patients()
    # print(json.dumps(report, indent=2, ensure_ascii=False))
    pass