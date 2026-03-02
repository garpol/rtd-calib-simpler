""" HAY QUE CAMBIAR EL FILTRO DE SIGMAS PARA EVITAR ERROR EN SET 57 
Clase Set - Agrupa múltiples runs del mismo set de calibración.

Responsabilidades:
- Agrupar runs por CalibSetNumber del LogFile.csv
- Filtrar runs inválidos (BAD, pre-calibración, etc.)
- Calcular constantes de calibración (dentro de cada set) promediando offsets de todos los runs
- Calcular errores asociados a los offsets (desviación estándar) de cada run dentro de cada set
- Guardar resultados en un archivo Excel 
"""
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from .utils import load_config # relative import
    from .run import Run
    from .logfile import Logfile
except ImportError:
    from utils import load_config # absolute import
    from run import Run
    from logfile import Logfile


class Set:
    """Agrupa y procesa múltiples runs de calibración."""
    
    def __init__(self, logfile_path: str = None, logfile_df: pd.DataFrame = None, config_path: str = None):
        """
        Crea una instancia de Set.
        
        Args:
            logfile_path: Ruta al archivo LogFile.csv
            logfile_df: DataFrame del logfile (alternativo a logfile_path)
            config_path: Ruta al archivo de configuración YAML
        """
        # Cargar logfile
        if logfile_df is not None:
            self.logfile = logfile_df
        elif logfile_path:
            lf = Logfile(filepath=logfile_path) # inicio clase Logfile
            self.logfile = lf.log_file # DataFrame del logfile
        else:
            # Ruta por defecto
            repo_root = Path(__file__).parents[1] # RTD_Calibration/
            default_path = repo_root / "data" / "LogFile.csv" # RTD_Calibration/data/LogFile.csv
            lf = Logfile(filepath=str(default_path)) #por defecto
            self.logfile = lf.log_file 
        
        # Cargar configuración
        self.config = load_config(config_path)
        
        # Diccionario para almacenar runs agrupados por set
        self.runs_by_set = {}
        
        # Resultados de calibración
        self.calibration_constants = {}
        self.calibration_errors = {}
    
    def group_runs(self, selected_sets=None):
        """
        Agrupa los runs por CalibSetNumber.
        
        Args:
            selected_sets: Lista de sets a procesar (None = todos)
        
        Notes:
            CalibSetNumber en logfile puede ser:
            - Numérico (como string): "0", "1", "2", ..., "63"
            - Especial (con texto): "FRAME_SET1", "FRAME_SET2", "RESIST_SET"
            
            Esta función convierte temporalmente a numérico (no modifica original)
            y filtra solo valores enteros positivos 1-63 (ignora set 0 y especiales).
        """
        print("\n=== Agrupando runs por CalibSetNumber ===")
        
        # Convertir CalibSetNumber a numérico TEMPORALMENTE (no modifica self.logfile)
        calib_set_numeric = pd.to_numeric(self.logfile["CalibSetNumber"], errors='coerce')
        
        # Obtener sets válidos: numéricos enteros entre 1-63
        valid_nums = calib_set_numeric.dropna()
        valid_sets = valid_nums[(valid_nums > 0) & (valid_nums % 1 == 0) & (valid_nums <= 63)]
        all_sets = sorted(valid_sets.unique())
        
        # Aplicar filtro de sets seleccionados si existe
        if selected_sets:
            all_sets = [s for s in all_sets if s in selected_sets]
        
        print(f"Sets a procesar: {all_sets}")
        
        # Palabras clave a excluir en filenames
        exclude_keywords = ['pre', 'st', 'lar']
        
        # Procesar cada set
        for set_num in all_sets:
            print(f"\nProcesando Set {set_num}")
            
            # Filtrar runs del set y crear copia 
            runs_df = self.logfile[calib_set_numeric == set_num].copy()
            
            # Aplicar exclusiones con operaciones pandas
            filename_mask = runs_df['Filename'].str.lower().apply(
                lambda x: isinstance(x, str) and not any(kw in x for kw in exclude_keywords)
            ) # True si no contiene keywords excluyentes
            valid_mask = filename_mask & (runs_df['Selection'] != 'BAD')
            
            # Crear instancias de Run para archivos válidos
            valid_runs = {}
            for filename in runs_df[valid_mask]['Filename']:
                try:
                    run = Run(filename, self.logfile, config=self.config)
                    valid_runs[filename] = run # Almacenar instancia de Run
                    print(f"  Incluido: {filename}")
                except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
                    print(f"  Error en {filename}: {e}")
            
            # Reportar exclusiones
            for filename in runs_df[~valid_mask]['Filename']:
                if isinstance(filename, str):
                    reason = 'keywords' if any(kw in filename.lower() for kw in exclude_keywords) else 'BAD'
                    print(f"  Excluido ({reason}): {filename}")
            
            if valid_runs:
                self.runs_by_set[set_num] = valid_runs
                print(f"  Total runs válidos: {len(valid_runs)}")
        
        print(f"\nTotal sets procesados: {len(self.runs_by_set)}")
    
    def calculate_calibration_constants(self, selected_sets=None):
        """
        Calcula matrices NxN de offsets entre sensores dentro del mismo set.
        
        NOTA IMPORTANTE:
        Estas constantes son PROVISIONALES (se calculan respecto de todos los sensores y no solo de los 'raised') 
        y se usan como input para la clase Tree.

        Tree es quien calcula las constantes de calibración FINALES de forma escalonada:
          - Encadena offsets a través de sensores 'raised' entre sets de diferentes rondas
          - Pondera múltiples caminos para cada sensor
          - Genera constantes globales respecto a una referencia absoluta (Set R3)
        
        Las matrices NxN aquí calculadas permiten a Tree acceder a cualquier par
        (sensor_i, sensor_j) necesario para el encadenamiento.
        
        Args:
            selected_sets: Lista de sets a procesar (None = todos)
        """
        print("\n=== Calculando matrices de offsets (provisionales) ===")
        
        sets_to_process = selected_sets if selected_sets else list(self.runs_by_set.keys()) # Todos los sets con runs
        
        for set_num in sets_to_process:
            if set_num not in self.runs_by_set:
                print(f"Advertencia: Set {set_num} no tiene runs cargados")
                continue
            
            print(f"\nSet {set_num}:")
            runs = self.runs_by_set[set_num] 
            
            # Recolectar offsets y errores de todos los runs
            offsets_list = []
            errors_list = []
            sensor_names = None  # Guardar nombres de sensores del primer run
            
            for filename, run in runs.items():
                try:
                    # calculate_offsets ahora retorna (offsets, errors)
                    offsets, errors = run.calculate_offsets()
                    offsets_list.append(offsets.values)
                    errors_list.append(errors.values)
                    
                    # Guardar nombres de sensores del primer run válido
                    if sensor_names is None:
                        sensor_names = list(offsets.index)
                    
                    print(f"  ✓ {filename}")
                except (AttributeError, ValueError, KeyError) as e:
                    print(f"  ✗ Error en {filename}: {e}")
            
            if not offsets_list:
                print(f"  Sin offsets válidos para set {set_num}")
                continue
            
            # Convertir a arrays
            offsets_array = np.array(offsets_list)  # shape: (n_runs, n_sensors, n_sensors)
            errors_array = np.array(errors_list)
            
            # Calcular promedio ponderado por inverso del error al cuadrado. Donde error = 0 o NaN, usar peso = 0
            weights = np.zeros_like(errors_array) # Inicializar pesos
            mask_valid = (errors_array > 0) & np.isfinite(errors_array) & np.isfinite(offsets_array) # Máscara para valores válidos 
            weights[mask_valid] = 1.0 / (errors_array[mask_valid] ** 2) # Inverso del error al cuadrado
            
            # Promedio ponderado
            weighted_sum = np.sum(offsets_array * weights, axis=0) # Suma ponderada de offsets 
            total_weights = np.sum(weights, axis=0) # Suma de pesos
            
            # Evitar división por cero con np.divide y obtener constantes de calibración 
            constants = np.divide(
                weighted_sum, 
                total_weights, 
                out=np.full_like(weighted_sum, np.nan),
                where=total_weights > 0
            ) # shape: (n_sensors, n_sensors)
            
            # Error = desviación estándar de los offsets
            errors = np.std(offsets_array, axis=0, ddof=1) # shape: (n_sensors, n_sensors)
            
            # Forzar diagonal a 0 (un sensor consigo mismo tiene offset=0 y error=0)
            np.fill_diagonal(constants, 0.0)
            np.fill_diagonal(errors, 0.0)
            
            # Convertir a DataFrames con nombres de sensores
            constants_df = pd.DataFrame(constants, index=sensor_names, columns=sensor_names)
            errors_df = pd.DataFrame(errors, index=sensor_names, columns=sensor_names)
            
            # Excluir sensores descartados y referencias del cálculo
            set_config = self.config.get('sensors', {}).get('sets', {}).get(set_num, {})
            discarded_sensors = set_config.get('discarded', [])
            reference_sensors = set_config.get('reference', [])
            excluded_sensors = discarded_sensors + reference_sensors
            
            if excluded_sensors and sensor_names:
                # Convertir a strings para comparar con nombres de sensores
                excluded_sensors_str = [str(s) for s in excluded_sensors]
                
                # Filtrar sensores que existen en la matriz
                sensors_to_exclude = [s for s in excluded_sensors_str if s in sensor_names]
                
                if sensors_to_exclude:
                    # Marcar filas y columnas de sensores excluidos como NaN
                    constants_df.loc[sensors_to_exclude, :] = np.nan
                    constants_df.loc[:, sensors_to_exclude] = np.nan
                    errors_df.loc[sensors_to_exclude, :] = np.nan
                    errors_df.loc[:, sensors_to_exclude] = np.nan
                    
                    print(f"  Sensores excluidos: {sensors_to_exclude}")
                    print(f"    - Referencias: {[s for s in sensors_to_exclude if int(s) in reference_sensors]}")
                    print(f"    - Descartados: {[s for s in sensors_to_exclude if int(s) in discarded_sensors]}")
            
            self.calibration_constants[set_num] = constants_df
            self.calibration_errors[set_num] = errors_df
            
            print(f"  Constantes calculadas: {constants_df.shape}")
            print(f"  Promedio de offsets: {np.nanmean(np.abs(constants)):.4e} K")
            print(f"  Error promedio: {np.nanmean(errors):.4e} K")
        
        print(f"\nTotal sets con constantes: {len(self.calibration_constants)}")
    
    def save_results(self, output_file="set_calibration_results.xlsx", save_csv=True):
        """
        Guarda los resultados en archivos Excel y CSV.
        
        Args:
            output_file: Nombre del archivo Excel de salida (se guardará en docs/)
            save_csv: Si True, también guarda cada matriz en CSV individual
        """
        if not self.calibration_constants:
            print("No hay constantes calculadas para guardar")
            return
        
        # Construir ruta hacia docs/
        from pathlib import Path
        docs_dir = Path(__file__).parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)  # Crear si no existe
        output_path = docs_dir / output_file
        
        print(f"\nGuardando resultados...")
        
        # Guardar Excel
        with pd.ExcelWriter(output_path) as writer:
            for set_num in sorted(self.calibration_constants.keys()):
                # Guardar constantes
                sheet_name = f"Set_{int(set_num)}_Constants"
                self.calibration_constants[set_num].to_excel(writer, sheet_name=sheet_name)
                
                # Guardar errores
                sheet_name = f"Set_{int(set_num)}_Errors"
                self.calibration_errors[set_num].to_excel(writer, sheet_name=sheet_name)
        
        print(f"✓ Excel guardado en {output_path}")
        
        # Guardar CSVs individuales
        if save_csv:
            for set_num in sorted(self.calibration_constants.keys()):
                # CSV de constantes
                csv_constants = docs_dir / f"set_{int(set_num)}_constants.csv"
                self.calibration_constants[set_num].to_csv(csv_constants)
                
                # CSV de errores
                csv_errors = docs_dir / f"set_{int(set_num)}_errors.csv"
                self.calibration_errors[set_num].to_csv(csv_errors)
            
            print(f"✓ CSVs guardados en {docs_dir}/ (set_N_constants.csv, set_N_errors.csv)")
    
    def get_summary(self):
        """
        Genera un resumen de los resultados.
        
        Returns:
            DataFrame con estadísticas por set, incluyendo info de sensores excluidos
        """
        summary = []
        for set_num in sorted(self.calibration_constants.keys()):
            constants = self.calibration_constants[set_num]
            errors = self.calibration_errors[set_num]
            
            # Obtener información de exclusión del config
            set_config = self.config.get('sensors', {}).get('sets', {}).get(set_num, {})
            n_references = len(set_config.get('reference', []))
            n_discarded = len(set_config.get('discarded', []))
            n_excluded = n_references + n_discarded
            n_valid = constants.shape[0] - n_excluded
            
            summary.append({
                "CalibSetNumber": set_num,
                "N_Sensors_Total": constants.shape[0],
                "N_Sensors_Valid": n_valid,
                "N_References": n_references,
                "N_Discarded": n_discarded,
                "N_Runs": len(self.runs_by_set[set_num]),
                "Mean_Offset_K": np.nanmean(np.abs(constants.values)),
                "Mean_Error_K": np.nanmean(errors.values),
                "Max_Offset_K": np.nanmax(np.abs(constants.values)),
                "Max_Error_K": np.nanmax(errors.values)
            })
        
        return pd.DataFrame(summary)
    
    def calculate_repeatability_histograms(self, selected_sets=None):
        """
        Calcula histogramas de sigmas de repetibilidad por ronda.
        
        Para cada sensor en un set, calcula la desviación estándar (sigma) del offset 
        entre ese sensor y el sensor de referencia del set, usando los 4 runs.
        
        Args:
            selected_sets: Lista de sets a procesar (None = todos)
            
        Returns:
            Tupla (all_data, filtered_data) donde:
            - all_data: Dict con todos los sensores {set_num: {'sensor_id': sigma_value, ...}}
            - filtered_data: Dict sin sensores descartados {set_num: {'sensor_id': sigma_value, ...}}
        """
        print("\n=== Calculando histogramas de repetibilidad por ronda ===")
        
        sets_to_process = selected_sets if selected_sets else list(self.runs_by_set.keys())
        repeatability_data_all = {}
        repeatability_data_filtered = {}
        
        for set_num in sets_to_process:
            if set_num not in self.runs_by_set:
                print(f"Advertencia: Set {set_num} no tiene runs cargados")
                continue
            
            print(f"\nSet {set_num}:")
            runs = self.runs_by_set[set_num]
            
            # Obtener configuración del set
            # NOTA: reference_sensors se usa SOLO para excluir esos sensores del output filtrado.
            # La referencia matemática es siempre el 2do sensor del mapping (ref=2), NO el config.
            set_config = self.config.get('sensors', {}).get('sets', {}).get(set_num, {})
            reference_sensors = set_config.get('reference', [])
            discarded_sensors = set_config.get('discarded', [])
            discarded_sensors_str = [str(s) for s in discarded_sensors]
            refs_str = [str(r) for r in reference_sensors]  # solo para excluir del output

            # --- PASO 1: Cargar todos los offsets de todos los runs ---
            all_run_offsets = {}
            for filename, run in runs.items():
                try:
                    offsets, _ = run.calculate_offsets()
                    all_run_offsets[filename] = offsets
                except Exception as e:
                    print(f"  ✗ Error cargando {filename}: {e}")

            if not all_run_offsets:
                print(f"  Sin datos válidos para set {set_num}")
                continue

            # --- PASO 2: Asignar referencia matemática por sensor ---
            # Igual que proyecto bueno: ref=2 (2do sensor del mapping) salvo si hay raised sensors.
            # refs_str se definió arriba y es SOLO para excluir del output.
            raised_sensors = set_config.get('raised', [])
            raised_str = [str(r) for r in raised_sensors]
            
            # Construir mapping de sensor_id -> posición de canal
            example_run = next(iter(all_run_offsets.values()))
            sensor_to_channel = {}
            channel_to_sensor = {}
            
            for idx, sid in enumerate(example_run.index):
                sensor_to_channel[str(sid)] = idx
                channel_to_sensor[idx] = str(sid)
            
            # Determinar referencia por sensor (igual que proyecto bueno)
            ref_por_sensor = {}
            
            if len(raised_str) >= 2:
                # Caso 1: Hay 2+ sensores raised → usar referencias dinámicas
                raised_channels = [sensor_to_channel.get(r) for r in raised_str if sensor_to_channel.get(r) is not None][:2]
                
                for sid in sensor_to_channel.keys():
                    ch = sensor_to_channel[sid]
                    if sid in raised_str:
                        # Si el sensor es raised, usa el otro raised como ref
                        other_raised = [channel_to_sensor[rc] for rc in raised_channels if channel_to_sensor[rc] != sid]
                        ref_por_sensor[sid] = other_raised[0] if other_raised else raised_str[0]
                    else:
                        # Sensor normal: usa el raised más cercano circularmente
                        distances = [
                            min(abs(ch - rc), 12 - abs(ch - rc))
                            for rc in raised_channels
                        ]
                        closest_idx = np.argmin(distances)
                        ref_por_sensor[sid] = channel_to_sensor[raised_channels[closest_idx]]
                
                print(f"  ℹ️  Referencias dinámicas: sensores raised {raised_str[:2]}")
            else:
                # Caso 2: Menos de 2 raised → usar canal fijo (igual que proyecto bueno con ref=2)
                # El proyecto bueno usa el segundo sensor del mapping como referencia fija
                ref_idx = 1  # Índice 1 = segundo sensor (ref=2 en base-1)
                fixed_ref = channel_to_sensor.get(ref_idx)
                
                if fixed_ref:
                    # Todos los sensores usan el segundo sensor del mapping como referencia
                    for sid in sensor_to_channel.keys():
                        ref_por_sensor[sid] = fixed_ref
                    print(f"  ℹ️  Usando canal fijo (ref=2): sensor {fixed_ref}")

            # --- PASO 3: Calcular sigma como std de MEDIAS POR RUN (igual que proyecto bueno) ---
            # Usa la referencia específica asignada a cada sensor (referencias dinámicas)
            sigma_by_sensor_all = {}
            sigma_by_sensor_filtered = {}
            
            # Construir lista de sensores
            all_sensors = set(sensor_to_channel.keys())
            
            for sensor_id in all_sensors:
                # Referencia específica para este sensor (siempre del mapping, nunca del config)
                ref_sensor = ref_por_sensor.get(sensor_id)
                
                # Calcular la media de offset por run
                run_means = []
                for filename, offsets in all_run_offsets.items():
                    if ref_sensor is None or ref_sensor not in offsets.columns or sensor_id not in offsets.index:
                        continue
                    val = offsets.loc[sensor_id, ref_sensor]
                    if np.isfinite(float(val)):
                        run_means.append(float(val))
                
                if len(run_means) < 2:
                    sigma_by_sensor_all[str(sensor_id)] = np.nan
                    if str(sensor_id) not in discarded_sensors_str and str(sensor_id) not in refs_str:
                        sigma_by_sensor_filtered[str(sensor_id)] = np.nan
                    continue
                
                run_means_arr = np.array(run_means)
                
                # Filtro IQR sobre las medias por run (igual que proyecto bueno)
                if len(run_means_arr) >= 4:
                    q1 = np.percentile(run_means_arr, 25)
                    q3 = np.percentile(run_means_arr, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    filtered_means = run_means_arr[(run_means_arr >= lower_bound) & (run_means_arr <= upper_bound)]
                    n_filtered = len(run_means_arr) - len(filtered_means)
                    if n_filtered > 0:
                        print(f"  ⚠️  Sensor {sensor_id}: {n_filtered} run(s) filtrado(s) por IQR "
                              f"[{lower_bound*1000:.2f}, {upper_bound*1000:.2f}] mK")
                    if len(filtered_means) >= 2:
                        sigma = np.std(filtered_means, ddof=1)
                    elif len(filtered_means) == 1:
                        sigma = np.nan
                    else:
                        sigma = np.std(run_means_arr, ddof=1)
                else:
                    sigma = np.std(run_means_arr, ddof=1)
                
                sigma_by_sensor_all[str(sensor_id)] = sigma
                
                # filtered: excluir sensores de referencia y descartados
                if str(sensor_id) not in discarded_sensors_str and str(sensor_id) not in refs_str:
                    sigma_by_sensor_filtered[str(sensor_id)] = sigma
            
            repeatability_data_all[set_num] = sigma_by_sensor_all
            repeatability_data_filtered[set_num] = sigma_by_sensor_filtered
            
            # Estadísticas del set
            valid_sigmas_all = [s for s in sigma_by_sensor_all.values() if np.isfinite(s)]
            valid_sigmas_filtered = [s for s in sigma_by_sensor_filtered.values() if np.isfinite(s)]
            
            if valid_sigmas_all:
                print(f"  Sensor de referencia: {ref_sensor}")
                print(f"  Sensores descartados: {discarded_sensors}")
                print(f"  Total sensores procesados: {len(sigma_by_sensor_all)}")
                print(f"  Sigmas válidas (todos): {len(valid_sigmas_all)}")
                print(f"  Sigmas válidas (sin descartados): {len(valid_sigmas_filtered)}")
                print(f"  Sigma media (todos): {np.mean(valid_sigmas_all)*1000:.3f} mK")
                print(f"  Sigma media (sin descartados): {np.mean(valid_sigmas_filtered)*1000:.3f} mK")
                print(f"  Sigma máxima (todos): {np.max(valid_sigmas_all)*1000:.3f} mK")
        
        print(f"\nTotal sets procesados: {len(repeatability_data_all)}")
        return repeatability_data_all, repeatability_data_filtered
