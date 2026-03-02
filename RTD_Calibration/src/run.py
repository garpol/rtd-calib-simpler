"""
Clase Run - Procesamiento de archivos individuales de calibración

Responsabilidades:
- Cargar archivos de temperaturas (.txt)
- Limpiar datos inválidos (NaNs, fuera de rango)
- Asociar sensores RTD a canales de medición
- Calcular offsets y sus errores entre todos los pares de sensores

Flujo de trabajo:
    1. Buscar archivo .txt en data/temperature_files/
    2. Parsear fechas y extraer canales de temperatura (channel_1 a channel_14)
    3. Filtrar valores fuera del rango válido (60-350°C por defecto)
    4. Detectar canales defectuosos (>40 NaNs por defecto)
    5. Asociar canales a IDs de sensores usando LogFile.csv
    6. Calcular offsets en ventana estable (20-40 min)
    7. Calcular errores RMS en la misma ventana

Para ejemplos de uso interactivo, ver: notebooks/RUN.ipynb

Configuración: Los parámetros se leen de config/config.yml (run_options)
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .utils import load_config, get_run_metadata
except ImportError:
    from utils import load_config, get_run_metadata


class Run:
    """Procesa un archivo de calibración individual."""
    
    def __init__(self, filename: str, logfile: pd.DataFrame, config: dict = None) -> None:
        """
        Crea una instancia de Run.
        
        Args:
            filename: Nombre del archivo de temperaturas (sin .txt)
            logfile: DataFrame con información de mapeo de sensores
            config: Diccionario con configuración (opcional, se carga de config.yml)
        """
        self.filename = filename
        self.logfile = logfile
        self.temperature_data = None
        self.sensor_mapping = None
        self.defective_channels = []
        
        # Cargar configuración
        if config is None:
            config = load_config() #funcion de utils.py
        self.config = config
        
        # Obtener parámetros de run_options 
        run_opts = self.config.get('run_options', {}) #run options en config.yml / utils 
        self.max_nan_threshold = run_opts.get('max_nan_threshold', 40) #umbral de NaNs para canales defectuosos 
        temp_range = run_opts.get('valid_temp_range', {}) #rango de temperaturas válidas
        self.temp_min = temp_range.get('min', 70) #60 grados por defecto por si no está en config.yml
        self.temp_max = temp_range.get('max', 80) #temperatura máxima por defecto 
        
        # Cargar automáticamente los datos al iniciar el objeto 
        self._load_and_process()
    
    def _load_and_process(self):
        """Carga y procesa el archivo de temperaturas."""
        try:
            self._load_temperature_file() #carga datos de temperatura
            self._associate_sensors() #asocia sensores RTD a canales
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e: # Manejo de errores
            print(f"Error procesando {self.filename}: {e}")
            self.temperature_data = pd.DataFrame() # DataFrame vacío en caso de error
    
    @property
    def metadata(self) -> dict:
        """
        Obtiene metadata completa del run desde el logfile (lazy loading).
        
        Returns:
            dict: Diccionario con información del run:
                - set_number: Número del set de calibración
                - ...
                - y más (ver utils.get_run_metadata)
        
        Notes:
            - Uso lazy: solo se calcula cuando se accede por primera vez
            - Reutiliza get_run_metadata() de utils.py
            - Útil para filtrar runs, agrupar por set, etc.
        """
        # Lazy loading: calcular solo si no existe, evita cálculos innecesarios.
        if not hasattr(self, '_metadata'): #Si no existe el atributo _metadata
            self._metadata = get_run_metadata(self.filename, self.logfile) #llamar a la función de utils.py para obtener metadata   
        return self._metadata
    
    def _load_temperature_file(self):
        """Busca y carga el archivo de temperaturas."""
        # Buscar archivo
        repo_root = Path(__file__).parents[1] # Raíz del repositorio
        search_path = repo_root / "data" / "temperature_files" # Carpeta de archivos de temperatura
        
        matches = glob.glob(str(search_path / "**" / f"{self.filename}.txt"), recursive=True) # Buscar recursivamente
        if not matches: # No se encontró ningún archivo
            raise FileNotFoundError(f"No se encontró {self.filename}.txt")
        
        filepath = matches[0] # Tomar el primer match encontrado
        print(f"Cargando: {filepath}")
        
        # Leer archivo (sin encabezado, separado por tabulaciones):
        df = pd.read_csv(filepath, sep="\t", header=None)
        
        # Nombrar columnas:
        cols = ["Date", "Time"] + [f"channel_{i}" for i in range(1, 15)] # fecha, hora, channel_1..channel_14
        df.columns = cols + list(df.columns[len(cols):]) # Mantener columnas adicionales si existen
        
        # Parsear fechas 
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], 
            errors="coerce",
            format="%m/%d/%Y %I:%M:%S %p"
        ) # Intentar formato con AM/PM, I es la hora en formato 12h y 'p' indica AM/PM
        if df["datetime"].isna().all(): # ¿TODAS las fechas son NaT (not a time)?
            df["datetime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"], 
                errors="coerce"
            ) # Intentar sin especificar formato (pandas infiere automáticamente) 
        
        # Extraer solo canales de temperatura
        temp_cols = [c for c in df.columns if c.startswith("channel_")] #channel_1..channel_14
        temps = df[temp_cols].copy() #copiar solo las columnas de temperatura
        temps.index = df["datetime"] # Usar datetime como índice
        
        # Filtrar valores inválidos (rango configurable desde config.yml)
        temps = temps.mask((temps < self.temp_min) | (temps > self.temp_max)) #poner NaN fuera de rango
        temps = temps.dropna(how="all") # Eliminar filas donde todos los canales son NaNs
        
        # Detectar canales defectuosos (umbral configurable desde config.yml)
        nan_counts = temps.isna().sum() #contar NaNs en cada columna
        self.defective_channels = nan_counts[nan_counts > self.max_nan_threshold].index.tolist() #canales con más NaNs que el umbral
        if self.defective_channels: # Si hay canales defectuosos
            print(f"  Canales defectuosos: {self.defective_channels}")
            temps[self.defective_channels] = np.nan # Marcar canales defectuosos como NaN 24/02
        
        self.temperature_data = temps
        print(f"  Datos cargados: {len(temps)} registros")
    
    def _associate_sensors(self):
        """Asocia los IDs de los sensores RTD a los canales 1-14 de temperatura."""
        if self.temperature_data is None or self.temperature_data.empty:
            return
        
        # Buscar filename en logfile
        match = self.logfile[self.logfile["Filename"] == self.filename] #filtrar filas con el filename dado
        if match.empty:
            raise ValueError(f"No se encontró {self.filename} en el logfile")
        
        # Extraer IDs de sensores (S1 a S20), habrá NaNs porque no hay tantos sensores
        sensor_cols = [f"S{i}" for i in range(1, 21)] 
        sensor_ids = match[sensor_cols].iloc[0].dropna().values # Extraer solo valores (IDs) no NaN en la fila correspondiente.
        
        # Convertir a strings de enteros, incluyendo check de NaN
        sensor_ids = [str(int(float(x))) for x in sensor_ids if pd.notna(x)] #float para evitar errores con tipos inesperados.
        
        # Mapear channel_1..channel_14 a sensor IDs
        channels = [f"channel_{i}" for i in range(1, 15)] #channel_1 a channel_14
        self.sensor_mapping = dict(zip(channels, sensor_ids[:14])) #mapear solo hasta 14 sensores, dict(zip) es una forma rápida de crear un diccionario a partir de dos listas
        
        # Renombrar columnas en temperature_data con los IDs de sensores
        self.temperature_data = self.temperature_data.rename(columns=self.sensor_mapping) #las keys(channels) del dict se renombran a values(sensor_ids)
        print(f"  Sensores asociados: {len(self.sensor_mapping)}")
    
    def calculate_offsets(self, time_start=20, time_end=40, error_time_start=20, error_time_end=40):
        """
        Calcula offsets entre sensores con sus errores asociados.
        
        Tanto offsets como errores se calculan en la misma ventana de tiempo (20-40 min por defecto)
        que corresponde al periodo estable de la calibración.
        
        Args:
            time_start: Inicio de ventana en minutos desde el comienzo. 
                       Default 20 min = tiempo de estabilización térmica.
            time_end: Fin de ventana en minutos desde el comienzo.
                     Default 40 min = fin del periodo estable.
            error_time_start: Inicio de ventana para errores (desde inicio).
                             Default 20 min = misma ventana que offsets.
            error_time_end: Fin de ventana para errores (desde inicio).
                           Default 40 min = misma ventana que offsets.
                     
        Returns:
            tuple: (offsets, errors)
                - offsets: DataFrame con offsets (sensor_i - sensor_j) en K
                - errors: DataFrame con errores RMS (± incertidumbre) en K
                
        Example:
            >>> offsets, errors = run.calculate_offsets()
            >>> print(f"Offset: {offsets.loc['48060', '48061']:.3f} ± {errors.loc['48060', '48061']:.3f} K")
            
        Note:
            El error RMS mide la variabilidad temporal de las diferencias respecto al offset promedio.
            Para visualización interactiva, ver notebooks/RUN_BUENO.ipynb
        """
        if self.temperature_data is None or self.temperature_data.empty:
            raise ValueError("No hay datos de temperatura")
        
        # 1. Calcular offsets en ventana estable (desde el inicio)
        t0 = self.temperature_data.index.min() + pd.Timedelta(minutes=time_start)
        t1 = self.temperature_data.index.min() + pd.Timedelta(minutes=time_end)
        offset_window = self.temperature_data.loc[t0:t1] #ventana de temperaturas para offsets
        
        if offset_window.empty:
            raise ValueError("Ventana de tiempo para offsets vacía")
        
        sensors = list(offset_window.columns) # Lista de sensores disponibles
        offsets = pd.DataFrame(0.0, index=sensors, columns=sensors) # Inicialización de DataFrame para offsets
        
        # Calcular offsets entre todos los pares de sensores
        for i, s1 in enumerate(sensors):
            for j, s2 in enumerate(sensors):
                if i == j:
                    offsets.loc[s1, s2] = 0.0 # Offset cero consigo mismo
                elif i < j:
                    diff = (offset_window[s1] - offset_window[s2]).mean() # Offset medio en la ventana
                    offsets.loc[s1, s2] = diff
                    offsets.loc[s2, s1] = -diff # Offset no simétrico
        
        # 2. Calcular errores RMS en la misma ventana (20-40 min desde inicio)
        t0_err = self.temperature_data.index.min() + pd.Timedelta(minutes=error_time_start)
        t1_err = self.temperature_data.index.min() + pd.Timedelta(minutes=error_time_end)
        error_window = self.temperature_data.loc[t0_err:t1_err] #ventana de temperaturas para errores
        
        errors = pd.DataFrame(0.0, index=sensors, columns=sensors) # Inicialización de DataFrame para errores
        
        # Calcular RMS de las diferencias respecto al offset medio
        for i, s1 in enumerate(sensors):
            for j, s2 in enumerate(sensors):
                if i == j:
                    errors.loc[s1, s2] = 0.0 # Error cero consigo mismo
                elif i < j:
                    mean_offset = offsets.loc[s1, s2] # Offset medio calculado previamente
                    diff = error_window[s1] - error_window[s2] # Diferencias de temperatura en la ventana temporal
                    rms = np.sqrt(((diff - mean_offset) ** 2).mean()) #a cada valor de la serie 'diff' le resto el offset medio, lo elevo al cuadrado, hago la media y saco la raíz cuadrada.
                    errors.loc[s1, s2] = rms 
                    errors.loc[s2, s1] = rms  # Error simétrico
        
        return offsets, errors # Devuelve ambos DataFrames
