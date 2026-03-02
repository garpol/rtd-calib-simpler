"""
Clase Tree para calcular constantes de calibración entre sets de diferentes rondas.

La calibración sigue una estructura jerárquica:
- Ronda 1: Sets base (3-48) con sensores calibrados contra referencias
- Ronda 2: Sets (49-56) que conectan múltiples sets de R1
- Ronda 3: Sets (57+) que conectan múltiples sets de R2

Los sensores "raised" (elevados) actúan como puentes entre rondas,
permitiendo calcular offsets encadenados.

TODO LIST:
-----------
1. [PENDIENTE] Deduplicar sensores en CSV final de calculate_all_offsets()
   - Actualmente: Si un sensor aparece en múltiples sets (ej: descartado en Set 3, 
     recuperado en Set 5), se generan múltiples filas en el DataFrame final
   - Solución propuesta: Implementar lógica de priorización:
     a) Priorizar Status: 'Calculado' > 'Referencia' > 'Descartado' > 'Sin conexión'
     b) Si hay múltiples 'Calculado', tomar el del set más alto (recuperado)
     c) Si empatan en set, tomar el de menor error
   - Implementación: Añadir deduplicación al final de calculate_all_offsets()
     antes de retornar el DataFrame
   - Ventaja: Cada sensor tendrá UNA constante única (comportamiento esperado físicamente)
   - Opción: Guardar CSV "_debug.csv" con duplicados para trazabilidad si es necesario
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import yaml
try:
    from .utils import load_config, propagate_error
except ImportError:
    from utils import load_config, propagate_error


class Tree:
    """
    Calcula constantes de calibración globales conectando sets de diferentes rondas
    a través de sensores raised que sirven como puentes.
    """
    
    def __init__(self, sets_dict: Dict[float, Any], config_path: Optional[str] = None):
        """
        Inicializa el árbol de calibración.
        
        Args:
            sets_dict: Diccionario {set_id: objeto_Set} con sets procesados
            config_path: Ruta al archivo config.yml (opcional)
        """
        self.sets = sets_dict
        self.config = load_config(config_path)
        
        # Almacenar offsets calculados: {(sensor_origen, sensor_destino): (offset, error)}
        self.offsets: Dict[Tuple[int, int], Tuple[float, float]] = {}
        
        # Clasificar sets por ronda
        self.sets_by_round: Dict[int, List[float]] = {1: [], 2: [], 3: []}
        self._classify_sets_by_round()
    
    def _find_parent_sets(self, target_set_id: float) -> List[float]:
        """
        Encuentra los sets "parent" de un set dado analizando qué sets tienen
        sensores raised que aparecen como sensores normales en el target_set.
        
        Solo retorna parents de la ronda INMEDIATAMENTE ANTERIOR (no salta rondas).
        
        Deriva la estructura del árbol automáticamente desde config.yml.
        
        Args:
            target_set_id: ID del set para el cual buscar parents
        
        Returns:
            Lista de IDs de sets parent (ronda inmediatamente anterior)
        """
        sets_config = self.config.get('sensors', {}).get('sets', {})
        target_config = sets_config.get(float(target_set_id), {})
        target_sensors = set(target_config.get('sensors', []))
        target_round = target_config.get('round', 0)
        
        # La ronda parent debe ser inmediatamente anterior
        parent_round = target_round - 1
        if parent_round < 1:
            return []  # No hay ronda anterior
        
        parents = []
        
        # Buscar sets de la ronda anterior cuyo 'raised' contenga sensores del target_set
        for set_id, set_config in sets_config.items():
            if set_id == target_set_id:
                continue
            
            # Filtrar solo sets de la ronda parent
            set_round = set_config.get('round', 0)
            if set_round != parent_round:
                continue
            
            raised_sensors = set(set_config.get('raised', []))
            
            # Si algún raised de este set está en el target, es un parent
            if raised_sensors & target_sensors:  # Intersección
                parents.append(float(set_id))
        
        return sorted(parents)
    
    def get_r1_sets_for_r3_set(self, r3_set_id: int) -> List[int]:
        """
        Obtiene la lista de sets R1 que están relacionados con un set R3.
        Deriva la estructura R1 → R2 → R3 automáticamente desde config.yml.
        
        Args:
            r3_set_id: ID del set de ronda 3
        
        Returns:
            Lista de IDs de sets de ronda 1 relacionados
        """
        # Obtener parents de R3 (sets R2)
        r2_sets = self._find_parent_sets(float(r3_set_id))
        
        # Para cada set R2, obtener sus parents (sets R1)
        r1_sets = []
        for r2_id in r2_sets:
            r1_from_r2 = self._find_parent_sets(r2_id)
            r1_sets.extend(r1_from_r2)
        
        return sorted(list(set(int(s) for s in r1_sets)))  # Eliminar duplicados y ordenar
    
    def _classify_sets_by_round(self):
        """Clasifica los sets según su ronda usando config.yml"""
        sets_config = self.config.get('sensors', {}).get('sets', {})
        
        for set_id in self.sets.keys():
            set_config = sets_config.get(float(set_id), {})
            round_num = set_config.get('round', 1) # Default a ronda 1 si no está especificado
            
            if round_num in self.sets_by_round:
                self.sets_by_round[round_num].append(set_id)
        
        for round_num in self.sets_by_round:
            self.sets_by_round[round_num].sort()
        
        print("\nSets clasificados por ronda:")
        for round_num, sets in self.sets_by_round.items():
            if sets:
                print(f"  Ronda {round_num}: {len(sets)} sets -> {sets}")
    
    # =========================================================================
    # HELPERS para lógica compleja
    # =========================================================================
    
    def _create_result_record(self, sensor: int, set_id: float, round_num: int,
                             offset: Optional[float] = None, error: Optional[float] = None,
                             status: str = 'Sin conexión') -> dict:
        """
        Crea un registro de resultado para agregar al DataFrame.
        Reduce duplicación de código en calculate_all_offsets_*.
        
        Args:
            sensor: ID del sensor
            set_id: ID del set
            round_num: Número de ronda
            offset: Offset calculado (None = NaN)
            error: Error calculado (None = NaN)
            status: Estado del cálculo ('Calculado', 'Sin conexión', 'Descartado', 'Referencia')
        
        Returns:
            Diccionario con estructura estandarizada para el DataFrame
        """
        return {
            'Sensor': sensor,
            'Set': set_id,
            'Round': round_num,
            'Constante_Calibracion_K': offset if offset is not None else np.nan,
            'Error_K': error if error is not None else np.nan,
            'Status': status
        }
    
    def _create_multipath_result_record(self, sensor: int, set_id: float, 
                                       n_paths: int = 0,
                                       first_offset: Optional[float] = None,
                                       first_error: Optional[float] = None,
                                       min_offset: Optional[float] = None,
                                       min_error: Optional[float] = None,
                                       weighted_offset: Optional[float] = None,
                                       weighted_error: Optional[float] = None,
                                       std_paths: Optional[float] = None,
                                       max_diff: Optional[float] = None,
                                       status: str = 'Sin conexión') -> dict:
        """
        Crea un registro de resultado con estadísticas multi-camino.
        Reduce duplicación en calculate_all_offsets_multi_path.
        
        Args:
            sensor: ID del sensor
            set_id: ID del set
            n_paths: Número de caminos encontrados
            first_offset: Offset del primer camino
            first_error: Error del primer camino
            min_offset: Offset del camino con menor error
            min_error: Error mínimo
            weighted_offset: Offset de la media ponderada
            weighted_error: Error de la media ponderada
            std_paths: Desviación estándar entre caminos
            max_diff: Diferencia máxima entre caminos
            status: Estado del cálculo
        
        Returns:
            Diccionario con estructura para DataFrame multi-camino
        """
        return {
            'Sensor': sensor,
            'Set': set_id,
            'Round': 1,
            'N_Caminos': n_paths,
            'Constante_Primer_Camino_K': first_offset if first_offset is not None else np.nan,
            'Error_Primer_Camino_K': first_error if first_error is not None else np.nan,
            'Constante_Min_Error_K': min_offset if min_offset is not None else np.nan,
            'Error_Min_K': min_error if min_error is not None else np.nan,
            'Constante_Media_Ponderada_K': weighted_offset if weighted_offset is not None else np.nan,
            'Error_Media_Ponderada_K': weighted_error if weighted_error is not None else np.nan,
            'Std_Entre_Caminos_K': std_paths if std_paths is not None else np.nan,
            'Max_Diff_Caminos_K': max_diff if max_diff is not None else np.nan,
            'Status': status
        }
    
    def _determine_r1_sets(self, reference_set: float, r1_sets_range: Optional[tuple] = None) -> list:
        """
        Determina qué sets R1 procesar según el modo (manual con rango o automático desde config.yml)
        
        Args:
            reference_set: Set de referencia R3
            r1_sets_range: Tupla (min, max) para modo manual, None para automático desde config.yml
        
        Returns:
            Lista de IDs de sets R1 a procesar
        """
        if r1_sets_range:
            # Modo manual: usar rango especificado
            print(f"  Modo manual: Limitando a sets R1 del {r1_sets_range[0]} al {r1_sets_range[1]}")
            r1_sets = self.sets_by_round.get(1, [])
            r1_sets = [s for s in r1_sets if r1_sets_range[0] <= s <= r1_sets_range[1]]
        else:
            # Modo automático: derivar estructura desde config.yml
            r1_sets = self.get_r1_sets_for_r3_set(int(reference_set))
            print(f"  Modo automático: Procesando {len(r1_sets)} sets R1 derivados desde config.yml")
            print(f"  Sets R1: {r1_sets}")
        return sorted(r1_sets)
    
    def _get_reference_info(self, reference_set: float) -> tuple:
        """
        Obtiene información del set de referencia
        
        Returns:
            Tupla (reference_sensor, ref_sensors_list)
        """
        sets_config = self.config.get('sensors', {}).get('sets', {})
        ref_config = sets_config.get(float(reference_set), {})
        ref_sensors = ref_config.get('sensors', [])
        reference_sensors_config = ref_config.get('reference', [])
        raised_sensors_config = ref_config.get('raised', [])
        discarded_sensors_config = ref_config.get('discarded', [])
        
        if not ref_sensors:
            print(f"Error: Set {reference_set} no tiene sensores definidos")
            return None, []
        
        # Elegir el primer sensor que NO sea reference, raised ni discarded
        # Queremos un sensor normal y válido del set
        excluded = set(reference_sensors_config + raised_sensors_config + discarded_sensors_config)
        
        reference_sensor = None
        for sensor in ref_sensors:
            if sensor not in excluded:
                reference_sensor = sensor
                break
        
        # Si todos están excluidos (caso raro), usar el primero de todos modos
        if reference_sensor is None:
            reference_sensor = ref_sensors[0]
            print(f"Advertencia: Todos los sensores del Set {reference_set} están marcados como reference/raised/discarded. Usando {reference_sensor}")
        
        return reference_sensor, ref_sensors
    
    def _print_calibration_summary(self, df: pd.DataFrame, title: str):
        """Imprime resumen de resultados de calibración"""
        if df.empty:
            print(f"\n{title}: Sin resultados")
            return
        
        calculated = df[df['Status'] == 'Calculado']
        discarded = df[df['Status'] == 'Descartado']
        no_connection = df[df['Status'] == 'Sin conexión']
        
        print(f"\n{title}:")
        print(f"  Total sensores: {len(df)}")
        print(f"  Calculados: {len(calculated)}")
        print(f"  Descartados: {len(discarded)}")
        print(f"  Sin conexión: {len(no_connection)}")
        
        # Solo imprimir estadísticas de error si existe la columna Error_K
        if len(calculated) > 0 and 'Error_K' in calculated.columns:
            print(f"  Error promedio: {calculated['Error_K'].mean():.4f} K")
            print(f"  Error máximo: {calculated['Error_K'].max():.4f} K")
    
    # =========================================================================
    # FIN HELPERS
    # =========================================================================
    
    def get_offset_within_set(self, set_id: float, sensor_from: int, sensor_to: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Obtiene el offset entre dos sensores dentro del mismo set.
        
        Args:
            set_id: ID del set
            sensor_from: ID del sensor origen
            sensor_to: ID del sensor destino
        
        Returns:
            Tupla (offset, error) o (None, None) si no existe
        """
        set_obj = self.sets.get(set_id)
        if set_obj is None:
            return None, None
        
        constants = set_obj.calibration_constants
        errors = set_obj.calibration_errors
        
        if constants is None or errors is None:
            return None, None
        
        # Convertir IDs a strings por consistencia
        sf, st = str(sensor_from), str(sensor_to)
        
        try:
            if sf in constants.index and st in constants.columns:
                offset = constants.loc[sf, st]
                error = errors.loc[sf, st]
                
                # Verificar que no sean NaN
                if pd.isna(offset) or pd.isna(error):
                    return None, None
                
                return offset, error
        except (KeyError, IndexError):
            pass
        
        return None, None
    
    def get_raised_sensors(self, set_id: float) -> List[int]:
        """
        Obtiene la lista de sensores raised de un set desde config.yml
        
        Args:
            set_id: ID del set
        
        Returns:
            Lista de IDs de sensores raised
        """
        sets_config = self.config.get('sensors', {}).get('sets', {})
        set_config = sets_config.get(float(set_id), {})
        raised = set_config.get('raised', [])
        return raised if raised else []
    
    def calculate_offset_between_sets(self, sensor_from: int, sensor_to: int, 
                                      set_from: float = None, set_to: float = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcula el offset entre dos sensores de diferentes sets usando sensores raised como puentes.
        
        Procedimiento:
        1. Encontrar en qué sets están los sensores (o usar los proporcionados)
        2. Buscar un sensor raised común entre esos sets
        3. Calcular: offset(sensor_from -> raised) + offset(raised -> sensor_to)
        
        Args:
            sensor_from: ID del sensor origen
            sensor_to: ID del sensor destino
            set_from: Set donde está sensor_from (opcional, se busca automáticamente)
            set_to: Set donde está sensor_to (opcional, se busca automáticamente)
        
        Returns:
            Tupla (offset, error) o (None, None) si no se puede calcular
        """
        if set_from is None:
            set_from = self._find_set_containing_sensor(sensor_from)
        if set_to is None:
            set_to = self._find_set_containing_sensor(sensor_to)
        
        if set_from is None or set_to is None:
            return None, None
        
        if set_from == set_to:
            return self.get_offset_within_set(set_from, sensor_from, sensor_to)
        
        # Buscar sensor raised común
        raised_from = set(self.get_raised_sensors(set_from))
        raised_to = set(self.get_raised_sensors(set_to))
        common_raised = raised_from & raised_to
        
        if not common_raised:
            return self._calculate_offset_chain(sensor_from, sensor_to, set_from, set_to)
        
        # Usar el primer sensor raised común
        bridge_sensor = list(common_raised)[0]
        
        offset1, error1 = self.get_offset_within_set(set_from, sensor_from, bridge_sensor)
        offset2, error2 = self.get_offset_within_set(set_to, bridge_sensor, sensor_to)
        
        if offset1 is None or offset2 is None:
            return None, None
        
        # Sumar offsets y propagar errores
        total_offset = offset1 + offset2
        total_error = propagate_error(error1, error2)
        
        return total_offset, total_error
    
    def _find_set_containing_sensor(self, sensor_id: int) -> Optional[float]:
        """
        Encuentra el set que contiene un sensor dado.
        
        Args:
            sensor_id: ID del sensor
        
        Returns:
            ID del set o None si no se encuentra
        """
        sets_config = self.config.get('sensors', {}).get('sets', {})
        
        for set_id in self.sets.keys():
            set_config = sets_config.get(float(set_id), {})
            sensors = set_config.get('sensors', [])
            
            if sensor_id in sensors:
                return set_id
        
        return None
    
    def _calculate_offset_chain(self, sensor_from: int, sensor_to: int,
                                set_from: float, set_to: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcula offset siguiendo una cadena de sensores raised entre múltiples sets.
        Usa BFS para encontrar el camino más corto a través de las rondas.
        
        Ejemplo: Sensor en R1 -> Raised_R1 -> Raised_R2 -> Raised_R3 -> Sensor objetivo
        
        Args:
            sensor_from: ID del sensor origen
            sensor_to: ID del sensor destino
            set_from: Set donde está sensor_from
            set_to: Set donde está sensor_to
        
        Returns:
            Tupla (offset, error) o (None, None) si no hay camino
        """
        from collections import deque
        
        # BFS para encontrar camino de sets
        # Estado: (set_actual, camino_sets)
        queue = deque([(set_from, [set_from])])
        visited = {set_from}
        
        while queue:
            current_set, path = queue.popleft()
            
            if current_set == set_to:
                # Encontramos el camino, ahora calcular offset total
                return self._calculate_offset_through_path(sensor_from, sensor_to, path)
            
            # Explorar sets conectados via sensores raised
            raised_current = set(self.get_raised_sensors(current_set))
            
            for next_set in self.sets.keys():
                if next_set in visited or next_set == current_set:
                    continue
                
                # Obtener sensores del siguiente set
                sets_config = self.config.get('sensors', {}).get('sets', {})
                next_sensors = set(sets_config.get(float(next_set), {}).get('sensors', []))
                raised_next = set(self.get_raised_sensors(next_set))
                
                # Verificar conexión: raised de current están en sensores o raised de next
                if raised_current & (next_sensors | raised_next):
                    visited.add(next_set)
                    queue.append((next_set, path + [next_set]))
        
        return None, None
    
    def _calculate_offset_through_path(self, sensor_from: int, sensor_to: int, 
                                       path: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcula offset siguiendo un camino de sets conectados.
        
        Args:
            sensor_from: Sensor inicial
            sensor_to: Sensor final
            path: Lista de set_ids que conectan sensor_from con sensor_to
        
        Returns:
            Tupla (offset_total, error_total)
        """
        if len(path) < 2:
            # Mismo set, offset directo
            return self.get_offset_within_set(path[0], sensor_from, sensor_to)
        
        total_offset = 0.0
        total_error_sq = 0.0
        current_sensor = sensor_from
        
        # Recorrer el camino
        for i in range(len(path) - 1):
            set_current = path[i]
            set_next = path[i + 1]
            
            # Encontrar sensor puente entre set_current y set_next
            raised_current = set(self.get_raised_sensors(set_current))
            raised_next = set(self.get_raised_sensors(set_next))
            
            # Obtener sensores del siguiente set
            sets_config = self.config.get('sensors', {}).get('sets', {})
            next_sensors = set(sets_config.get(float(set_next), {}).get('sensors', []))
            
            # Bridge: raised de current que están en raised o sensores de next
            bridge_sensors = raised_current & (raised_next | next_sensors)
            
            if not bridge_sensors:
                return None, None
            
            bridge = list(bridge_sensors)[0]
            
            # Offset: current_sensor -> bridge (dentro de set_current)
            offset1, error1 = self.get_offset_within_set(set_current, current_sensor, bridge)
            
            if offset1 is None:
                return None, None
            
            total_offset += offset1
            total_error_sq += error1**2
            
            # Para el siguiente paso, current_sensor es el bridge
            current_sensor = bridge
        
        # Último paso: bridge -> sensor_to (dentro de set_to)
        final_set = path[-1]
        offset_final, error_final = self.get_offset_within_set(final_set, current_sensor, sensor_to)
        
        if offset_final is None:
            return None, None
        
        total_offset += offset_final
        total_error_sq += error_final**2
        
        return total_offset, np.sqrt(total_error_sq)
    
    def _calculate_chained_offset_r1(self, sensor: int, set_r1: float, 
                                      raised_r1: list, reference_sensor: int, 
                                      reference_set: float, 
                                      return_steps: bool = False) -> tuple:
        """
        Calcula offset encadenado para sensor de R1:
        Sensor_R1 → Raised_R1 (en R1) → Raised_R2 (en R2) → Referencia_R3 (en R3)
        
        Args:
            sensor: Sensor de R1 a calibrar
            set_r1: Set de R1 donde está el sensor
            raised_r1: Lista de raised del set R1
            reference_sensor: Sensor de referencia absoluta en R3
            reference_set: Set de referencia (R3)
            return_steps: Si True, retorna también los pasos intermedios
        
        Returns:
            Si return_steps=False: (offset, error) o (None, None) si falla
            Si return_steps=True: (offset, error, steps_dict) donde steps_dict contiene detalles de cada paso
        """
        # Intentar con todos los raised disponibles
        if not raised_r1:
            print(f"  ✗ Sensor {sensor} Set {set_r1}: No hay raised en el set")
            if return_steps:
                return None, None, None
            return None, None
        
        sets_config = self.config.get('sensors', {}).get('sets', {})
        r2_sets = self.sets_by_round.get(2, [])
        r3_sensors = sets_config.get(float(reference_set), {}).get('sensors', [])
        
        failures = []
        
        # Probar cada raised hasta encontrar uno que funcione
        for raised_local in raised_r1:
            # Paso 1: Sensor → Raised_R1 dentro del set R1
            offset_1, error_1 = self.get_offset_within_set(set_r1, sensor, raised_local)
            
            if offset_1 is None or pd.isna(offset_1):
                failures.append(f"Paso1: {sensor}→{raised_local} en Set{set_r1} = None/NaN")
                continue
            
            if error_1 is None or pd.isna(error_1):
                error_1 = 0.0
            
            # Paso 2: Encontrar en qué set R2 aparece raised_local
            set_r2 = None
            for r2_id in r2_sets:
                r2_config = sets_config.get(float(r2_id), {})
                if raised_local in r2_config.get('sensors', []):
                    set_r2 = r2_id
                    break
            
            if set_r2 is None:
                failures.append(f"Paso2: Raised {raised_local} no está en ningún Set R2")
                continue
            
            # Paso 3: Dentro de set R2, buscar raised que esté en R3
            r2_raised = sets_config.get(float(set_r2), {}).get('raised', [])
            
            # Encontrar bridge entre R2 y R3
            bridge_r2_r3 = None
            for r2_r in r2_raised:
                if r2_r in r3_sensors:
                    bridge_r2_r3 = r2_r
                    break
            
            if bridge_r2_r3 is None:
                failures.append(f"Paso3: Set R2 {set_r2} no tiene bridge a R3")
                continue
            
            # Offset: raised_local → bridge_r2_r3 en set R2
            # Caso especial: si son el mismo sensor, offset = 0
            if raised_local == bridge_r2_r3:
                offset_2 = 0.0
                error_2 = 0.0
            else:
                offset_2, error_2 = self.get_offset_within_set(set_r2, raised_local, bridge_r2_r3)
                
                if offset_2 is None or pd.isna(offset_2):
                    failures.append(f"Paso4: {raised_local}→{bridge_r2_r3} en Set{set_r2} = None/NaN")
                    continue
                
                if error_2 is None or pd.isna(error_2):
                    error_2 = 0.0
            
            # Paso 4: bridge_r2_r3 → reference_sensor en set R3
            # Caso especial: si el bridge ES el reference_sensor, offset = 0
            if bridge_r2_r3 == reference_sensor:
                offset_3 = 0.0
                error_3 = 0.0
            else:
                offset_3, error_3 = self.get_offset_within_set(reference_set, bridge_r2_r3, reference_sensor)
                
                if offset_3 is None or pd.isna(offset_3):
                    failures.append(f"Paso5: {bridge_r2_r3}→{reference_sensor} en Set{reference_set} = None/NaN")
                    continue
                
                if error_3 is None or pd.isna(error_3):
                    error_3 = 0.0
            
            # ¡Éxito! Encadenar offsets
            total_offset = offset_1 + offset_2 + offset_3
            total_error = propagate_error(error_1, error_2, error_3)
            
            if return_steps:
                steps = {
                    'Paso1_Sensor_from': sensor,
                    'Paso1_Sensor_to': raised_local,
                    'Paso1_Set': set_r1,
                    'Paso1_Offset_K': offset_1,
                    'Paso1_Error_K': error_1,
                    'Paso2_Sensor_from': raised_local,
                    'Paso2_Sensor_to': bridge_r2_r3,
                    'Paso2_Set': set_r2,
                    'Paso2_Offset_K': offset_2,
                    'Paso2_Error_K': error_2,
                    'Paso3_Sensor_from': bridge_r2_r3,
                    'Paso3_Sensor_to': reference_sensor,
                    'Paso3_Set': reference_set,
                    'Paso3_Offset_K': offset_3,
                    'Paso3_Error_K': error_3
                }
                return total_offset, total_error, steps
            
            return total_offset, total_error
        
        # Ningún raised funcionó - mostrar por qué
        print(f"  Sensor {sensor} Set {set_r1}: {'; '.join(failures[:2])}")
        if return_steps:
            return None, None, None
        return None, None

    def calculate_all_offsets(self, reference_set: Optional[float] = None, 
                             r1_sets_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        Calcula offsets de todos los sensores de Ronda 1 respecto a la referencia absoluta.
        
        Lógica:
        - Deriva la estructura desde config.yml para determinar qué sets R1 procesar según el set R3
        - Para cada sensor: calcular offset respecto a su raised local
        - Encadenar: Sensor_R1 → Raised_R1 → Raised_R2 → Referencia_R3
        - Todos los sensores deben tener conexión (path garantizado)
        
        Args:
            reference_set: ID del set de referencia R3 (por defecto el de mayor ronda)
            r1_sets_range: Tupla (min, max) para limitar sets R1 a procesar. 
                          Si no se especifica, deriva sets R1 desde config.yml
        
        Returns:
            DataFrame con offsets encadenados
        """
        if reference_set is None:
            reference_set = self._get_reference_set()
        
        print(f"\nCalculando offsets respecto a Set {reference_set} (Ronda 3)")
        
        # Determinar qué sets R1 procesar usando helper
        r1_sets = self._determine_r1_sets(reference_set, r1_sets_range)
        
        # Obtener información de referencia usando helper
        reference_sensor, _ = self._get_reference_info(reference_set)
        if reference_sensor is None:
            return pd.DataFrame()
        
        print(f"  Referencia absoluta: Sensor {reference_sensor}")
        
        results = []
        print(f"  Procesando {len(r1_sets)} sets de Ronda 1")
        
        total_sensors = 0
        for set_id in sorted(r1_sets):
            # Obtener configuración del set desde config
            sets_config = self.config.get('sensors', {}).get('sets', {})
            set_config = sets_config.get(float(set_id), {})
            set_sensors = set_config.get('sensors', [])
            set_raised = set_config.get('raised', [])
            set_discarded = set_config.get('discarded', [])
            
            if not set_raised:
                print(f"  Advertencia: Set {set_id} no tiene sensores raised")
                continue
            
            # Para cada sensor normal del set R1
            
            for sensor in set_sensors:
                total_sensors += 1
                
                # Verificar si el sensor está descartado
                if sensor in set_discarded:
                    results.append(self._create_result_record(sensor, set_id, 1, status='Sensor descartado'))
                    continue
                
                # Si es un sensor raised, calibrarlo respecto al OTRO raised del mismo set
                if sensor in set_raised:
                    # Obtener el otro sensor raised como referencia local
                    other_raised = [r for r in set_raised if r != sensor]
                    
                    if not other_raised:
                        # Solo hay 1 raised en el set, no se puede calibrar
                        results.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    # Calibrar raised respecto al otro raised (dentro del mismo set)
                    ref_raised = other_raised[0]
                    offset_raised, error_raised = self.get_offset_within_set(set_id, sensor, ref_raised)
                    
                    if offset_raised is None:
                        results.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    # Ahora encadenar: ref_raised → raised_R2 → ref_R3
                    offset_chain, error_chain = self._calculate_chained_offset_r1(
                        ref_raised, set_id, set_raised, reference_sensor, reference_set
                    )
                    
                    if offset_chain is None:
                        results.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    # Offset total: sensor→ref_raised + ref_raised→referencia
                    total_offset = offset_raised + offset_chain
                    total_error = propagate_error(error_raised, error_chain)
                    
                    results.append(self._create_result_record(sensor, set_id, 1, 
                                                              total_offset, total_error, 'Calculado'))
                    continue
                
                # Calcular offset encadenado: sensor → raised_R1 → raised_R2 → ref_R3
                offset, error = self._calculate_chained_offset_r1(
                    sensor, set_id, set_raised, reference_sensor, reference_set
                )
                
                if offset is not None:
                    results.append(self._create_result_record(sensor, set_id, 1, 
                                                              offset, error, 'Calculado'))
                else:
                    results.append(self._create_result_record(sensor, set_id, 1))
        
        # Agregar la referencia absoluta
        results.append(self._create_result_record(reference_sensor, reference_set, 3, 
                                                  0.0, 0.0, 'Referencia'))
        
        df = pd.DataFrame(results)
        df = df.sort_values(['Set', 'Sensor'])
        
        # Usar helper para imprimir resumen
        self._print_calibration_summary(df, "=== RESULTADOS ===")
        
        # Estadísticas adicionales específicas de este método
        calculated = df[df['Status'] == 'Calculado']
        if len(calculated) > 0:
            print("\nEstadísticas de offsets:")
            print(f"  Promedio: {calculated['Constante_Calibracion_K'].mean():.4f} K")
            print(f"  Desv. Std: {calculated['Constante_Calibracion_K'].std():.4f} K")
            print(f"  Min: {calculated['Constante_Calibracion_K'].min():.4f} K")
            print(f"  Max: {calculated['Constante_Calibracion_K'].max():.4f} K")
        
        print(f"Total sensores R1 procesados: {total_sensors}")
        
        return df
    
    def calculate_all_offsets_with_steps(self, reference_set: Optional[float] = None, 
                                         r1_sets_range: Optional[tuple] = None) -> tuple:
        """
        Calcula offsets y genera CSV con pasos intermedios detallados.
        
        Args:
            reference_set: ID del set de referencia R3
            r1_sets_range: Tupla (min, max) para limitar sets R1 a procesar
        
        Returns:
            (df_main, df_steps): DataFrame principal y DataFrame con pasos detallados
        """
        if reference_set is None:
            reference_set = self._get_reference_set()
        
        print(f"\nCalculando offsets CON PASOS DETALLADOS respecto a Set {reference_set}")
        
        # Determinar sets R1 usando helper
        r1_sets = self._determine_r1_sets(reference_set, r1_sets_range)
        print(f"  Procesando {len(r1_sets)} sets de Ronda 1")
        
        # Obtener configuración de referencia usando helper
        reference_sensor, _ = self._get_reference_info(reference_set)
        if reference_sensor is None:
            return pd.DataFrame(), pd.DataFrame()
        print(f"  Referencia absoluta: Sensor {reference_sensor}")
        
        results_main = []
        results_steps = []
        
        for set_id in sorted(r1_sets):
            # Obtener configuración del set desde config
            sets_config = self.config.get('sensors', {}).get('sets', {})
            set_config = sets_config.get(float(set_id), {})
            set_sensors = set_config.get('sensors', [])
            set_raised = set_config.get('raised', [])
            set_discarded = set_config.get('discarded', [])
            
            if not set_raised:
                continue
            
            for sensor in set_sensors:
                # Verificar si está descartado
                if sensor in set_discarded:
                    results_main.append({
                        'Sensor': sensor,
                        'Set': set_id,
                        'Round': 1,
                        'Constante_Calibracion_K': np.nan,
                        'Error_K': np.nan,
                        'Status': 'Sensor descartado'
                    })
                    continue
                
                # Si es raised, calibrar respecto al otro raised del mismo set
                if sensor in set_raised:
                    other_raised = [r for r in set_raised if r != sensor]
                    
                    if not other_raised:
                        results_main.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    ref_raised = other_raised[0]
                    
                    # Paso 0: sensor_raised → otro_raised (dentro del mismo set)
                    offset_0, error_0 = self.get_offset_within_set(set_id, sensor, ref_raised)
                    
                    if offset_0 is None:
                        results_main.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    # Para raised sensors, calculamos la cadena DESDE ref_raised pero empezando en R2
                    # (no necesitamos Paso1 dentro del mismo set R1 porque ya hicimos Paso0)
                    
                    # Encontrar ref_raised en R2
                    set_r2 = None
                    sets_config = self.config.get('sensors', {}).get('sets', {})
                    r2_sets = self.sets_by_round.get(2, [])
                    for r2_id in r2_sets:
                        r2_config = sets_config.get(float(r2_id), {})
                        if ref_raised in r2_config.get('sensors', []):
                            set_r2 = r2_id
                            break
                    
                    if set_r2 is None:
                        print(f"  ✗ Sensor {sensor} (raised): ref_raised {ref_raised} no está en ningún Set R2")
                        results_main.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    # Encontrar bridge de R2 a R3
                    r2_raised = sets_config.get(float(set_r2), {}).get('raised', [])
                    r3_sensors = sets_config.get(float(reference_set), {}).get('sensors', [])
                    bridge_r2_r3 = None
                    for r2_r in r2_raised:
                        if r2_r in r3_sensors:
                            bridge_r2_r3 = r2_r
                            break
                    
                    if bridge_r2_r3 is None:
                        print(f"  ✗ Sensor {sensor} (raised): Set R2 {set_r2} no tiene bridge a R3")
                        results_main.append(self._create_result_record(sensor, set_id, 1))
                        continue
                    
                    # Paso 1 (R2): ref_raised → bridge_r2_r3 en Set R2
                    if ref_raised == bridge_r2_r3:
                        offset_1 = 0.0
                        error_1 = 0.0
                    else:
                        offset_1, error_1 = self.get_offset_within_set(set_r2, ref_raised, bridge_r2_r3)
                        if offset_1 is None or pd.isna(offset_1):
                            print(f"  ✗ Sensor {sensor} (raised): {ref_raised}→{bridge_r2_r3} en Set{set_r2} = None/NaN")
                            results_main.append(self._create_result_record(sensor, set_id, 1))
                            continue
                        if error_1 is None or pd.isna(error_1):
                            error_1 = 0.0
                    
                    # Paso 2 (R3): bridge_r2_r3 → reference_sensor en Set R3
                    if bridge_r2_r3 == reference_sensor:
                        offset_2 = 0.0
                        error_2 = 0.0
                    else:
                        offset_2, error_2 = self.get_offset_within_set(reference_set, bridge_r2_r3, reference_sensor)
                        if offset_2 is None or pd.isna(offset_2):
                            print(f"  ✗ Sensor {sensor} (raised): {bridge_r2_r3}→{reference_sensor} en Set{reference_set} = None/NaN")
                            results_main.append(self._create_result_record(sensor, set_id, 1))
                            continue
                        if error_2 is None or pd.isna(error_2):
                            error_2 = 0.0
                    
                    # Total: offset_0 + offset_1 + offset_2
                    total_offset = offset_0 + offset_1 + offset_2
                    total_error = propagate_error(error_0, error_1, error_2)
                    
                    results_main.append(self._create_result_record(sensor, set_id, 1, 
                                                                   total_offset, total_error, 'Calculado'))
                    
                    # Guardar pasos simplificados (sin loop back)
                    steps_row = {
                        'Sensor': sensor,
                        'Set': set_id,
                        'Round': 1,
                        'Constante_Calibracion_K': total_offset,
                        'Error_Total_K': total_error,
                        'Paso0_Sensor_from': sensor,
                        'Paso0_Sensor_to': ref_raised,
                        'Paso0_Set': set_id,
                        'Paso0_Offset_K': offset_0,
                        'Paso0_Error_K': error_0,
                        'Paso1_Sensor_from': ref_raised,
                        'Paso1_Sensor_to': bridge_r2_r3,
                        'Paso1_Set': set_r2,
                        'Paso1_Offset_K': offset_1,
                        'Paso1_Error_K': error_1,
                        'Paso2_Sensor_from': bridge_r2_r3,
                        'Paso2_Sensor_to': reference_sensor,
                        'Paso2_Set': reference_set,
                        'Paso2_Offset_K': offset_2,
                        'Paso2_Error_K': error_2,
                        'Paso3_Sensor_from': None,
                        'Paso3_Sensor_to': None,
                        'Paso3_Set': None,
                        'Paso3_Offset_K': None,
                        'Paso3_Error_K': None
                    }
                    results_steps.append(steps_row)
                    continue
                
                # Verificar si está descartado
                if sensor in set_discarded:
                    results_main.append(self._create_result_record(sensor, set_id, 1, status='Sensor descartado'))
                    # No añadir pasos para descartados
                    continue
                
                # Calcular con pasos detallados
                result = self._calculate_chained_offset_r1(
                    sensor, set_id, set_raised, reference_sensor, reference_set,
                    return_steps=True
                )
                
                if result[0] is not None:
                    offset, error, steps = result
                    
                    results_main.append(self._create_result_record(sensor, set_id, 1, 
                                                                   offset, error, 'Calculado'))
                    
                    # Añadir fila con pasos
                    steps_row = {
                        'Sensor': sensor,
                        'Set': set_id,
                        'Round': 1,
                        'Constante_Calibracion_K': offset,
                        'Error_Total_K': error
                    }
                    steps_row.update(steps)
                    results_steps.append(steps_row)
                else:
                    results_main.append(self._create_result_record(sensor, set_id, 1))
        
        # Agregar referencia absoluta
        results_main.append(self._create_result_record(reference_sensor, reference_set, 3, 
                                                       0.0, 0.0, 'Referencia'))
        
        df_main = pd.DataFrame(results_main).sort_values(['Set', 'Sensor'])
        df_steps = pd.DataFrame(results_steps).sort_values(['Set', 'Sensor'])
        
        # Imprimir resumen usando helper
        self._print_calibration_summary(df_main, "=== RESULTADOS ===")
        print(f"CSV principal: {len(df_main)} filas")
        print(f"CSV pasos: {len(df_steps)} filas")
        
        return df_main, df_steps
    
    def _explore_all_paths(self, sensor: int, set_r1: float, 
                          raised_r1: list, reference_sensor: int, 
                          reference_set: float) -> list:
        """
        Explora TODOS los caminos posibles para un sensor y retorna lista de resultados válidos.
        
        Args:
            sensor: Sensor de R1 a calibrar
            set_r1: Set de R1 donde está el sensor
            raised_r1: Lista de raised del set R1
            reference_sensor: Sensor de referencia absoluta en R3
            reference_set: Set de referencia (R3)
        
        Returns:
            Lista de tuplas: [(offset, error, path_description), ...]
            donde path_description = {raised_used, set_r2, bridge_used}
        """
        if not raised_r1:
            return []
        
        sets_config = self.config.get('sensors', {}).get('sets', {})
        r2_sets = self.sets_by_round.get(2, [])
        r3_sensors = sets_config.get(float(reference_set), {}).get('sensors', [])
        
        valid_paths = []
        
        # Probar CADA raised de R1
        for raised_local in raised_r1:
            # Paso 1: Sensor → Raised_R1 dentro del set R1
            offset_1, error_1 = self.get_offset_within_set(set_r1, sensor, raised_local)
            
            if offset_1 is None or pd.isna(offset_1):
                continue
            
            if error_1 is None or pd.isna(error_1):
                error_1 = 0.0
            
            # Paso 2: Encontrar en qué set R2 aparece raised_local
            set_r2 = None
            for r2_id in r2_sets:
                r2_config = sets_config.get(float(r2_id), {})
                if raised_local in r2_config.get('sensors', []):
                    set_r2 = r2_id
                    break
            
            if set_r2 is None:
                continue
            
            # Paso 3: Buscar TODOS los bridges posibles entre R2 y R3
            r2_raised = sets_config.get(float(set_r2), {}).get('raised', [])
            
            for bridge_r2_r3 in r2_raised:
                if bridge_r2_r3 not in r3_sensors:
                    continue
                
                # Offset: raised_local → bridge_r2_r3 en set R2
                # Caso especial: si son el mismo sensor, offset = 0
                if raised_local == bridge_r2_r3:
                    offset_2 = 0.0
                    error_2 = 0.0
                else:
                    offset_2, error_2 = self.get_offset_within_set(set_r2, raised_local, bridge_r2_r3)
                    
                    if offset_2 is None or pd.isna(offset_2):
                        continue
                    
                    if error_2 is None or pd.isna(error_2):
                        error_2 = 0.0
                
                # Paso 4: bridge_r2_r3 → reference_sensor en set R3
                if bridge_r2_r3 == reference_sensor:
                    offset_3 = 0.0
                    error_3 = 0.0
                else:
                    offset_3, error_3 = self.get_offset_within_set(reference_set, bridge_r2_r3, reference_sensor)
                    
                    if offset_3 is None or pd.isna(offset_3):
                        continue
                    
                    if error_3 is None or pd.isna(error_3):
                        error_3 = 0.0
                
                # Camino válido encontrado
                total_offset = offset_1 + offset_2 + offset_3
                total_error = propagate_error(error_1, error_2, error_3)
                
                path_info = {
                    'raised_r1': raised_local,
                    'set_r2': set_r2,
                    'bridge_r2_r3': bridge_r2_r3,
                    'offset_1': offset_1,
                    'error_1': error_1,
                    'offset_2': offset_2,
                    'error_2': error_2,
                    'offset_3': offset_3,
                    'error_3': error_3
                }
                
                valid_paths.append((total_offset, total_error, path_info))
        
        return valid_paths
    
    def calculate_all_offsets_multi_path(self, reference_set: Optional[float] = None, 
                                         r1_sets_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        Analiza todos los caminos posibles para cada sensor y calcula:
        1. Constante por camino de error mínimo
        2. Constante por media ponderada (1/error²)
        3. Estadísticas de variabilidad entre caminos
        
        Args:
            reference_set: ID del set de referencia R3
            r1_sets_range: Tupla (min, max) para limitar sets R1
        
        Returns:
            DataFrame con análisis multi-camino
        """
        if reference_set is None:
            reference_set = self._get_reference_set()
        
        print("\n=== ANÁLISIS MULTI-CAMINO ===")
        print("Explorando TODOS los caminos posibles para cada sensor\n")
        
        # Determinar sets R1 usando helper
        r1_sets = self._determine_r1_sets(reference_set, r1_sets_range)
        
        # Obtener referencia usando helper
        reference_sensor, _ = self._get_reference_info(reference_set)
        if reference_sensor is None:
            return pd.DataFrame()
        
        print(f"Referencia absoluta: Sensor {reference_sensor} (Set {reference_set})")
        print(f"Procesando {len(r1_sets)} sets de Ronda 1\n")
        
        results = []
        
        for set_id in sorted(r1_sets):
            # Obtener configuración del set desde config
            sets_config = self.config.get('sensors', {}).get('sets', {})
            set_config = sets_config.get(float(set_id), {})
            set_sensors = set_config.get('sensors', [])
            set_raised = set_config.get('raised', [])
            set_discarded = set_config.get('discarded', [])
            
            if not set_raised:
                continue
            
            for sensor in set_sensors:
                # Verificar si está descartado
                if sensor in set_discarded:
                    results.append(self._create_multipath_result_record(sensor, set_id, 
                                                                        status='Sensor descartado'))
                    continue
                
                # Si es raised, calibrar respecto al otro raised
                if sensor in set_raised:
                    other_raised = [r for r in set_raised if r != sensor]
                    
                    if not other_raised:
                        results.append(self._create_multipath_result_record(sensor, set_id))
                        continue
                    
                    ref_raised = other_raised[0]
                    
                    # Paso 0: sensor→otro_raised
                    offset_0, error_0 = self.get_offset_within_set(set_id, sensor, ref_raised)
                    
                    if offset_0 is None:
                        results.append(self._create_multipath_result_record(sensor, set_id))
                        continue
                    
                    # Explorar caminos para ref_raised
                    paths_ref = self._explore_all_paths(
                        ref_raised, set_id, set_raised, reference_sensor, reference_set
                    )
                    
                    if not paths_ref:
                        results.append(self._create_multipath_result_record(sensor, set_id))
                        continue
                    
                    # Aplicar offset_0 a todos los caminos
                    all_paths = []
                    for offset_ref, error_ref, path_info in paths_ref:
                        total_offset = offset_0 + offset_ref
                        total_error = propagate_error(error_0, error_ref)
                        all_paths.append((total_offset, total_error, path_info))
                    
                    # Calcular estadísticas
                    offsets = np.array([p[0] for p in all_paths])
                    errors = np.array([p[1] for p in all_paths])
                    
                    # Estrategia 1: Primer camino
                    first_offset, first_error = all_paths[0][0], all_paths[0][1]
                    
                    # Estrategia 2: Mínimo error
                    min_idx = np.argmin(errors)
                    min_offset, min_error = all_paths[min_idx][0], all_paths[min_idx][1]
                    
                    # Estrategia 3: Media ponderada
                    # Pesos: w_i = 1/σ_i²
                    weights_unnorm = 1.0 / (errors**2 + 1e-10)
                    # Media ponderada: μ = Σ(x_i × w_i) / Σ(w_i)
                    weighted_offset = np.sum(offsets * weights_unnorm) / np.sum(weights_unnorm)
                    # Error de la media ponderada: σ = 1/√(Σ(w_i)) = 1/√(Σ(1/σ_i²))
                    weighted_error = 1.0 / np.sqrt(np.sum(weights_unnorm))
                    
                    results.append(self._create_multipath_result_record(
                        sensor, set_id, len(all_paths),
                        first_offset, first_error,
                        min_offset, min_error,
                        weighted_offset, weighted_error,
                        offsets.std(), offsets.max() - offsets.min(),
                        'Calculado'
                    ))
                    continue
                
                # Explorar TODOS los caminos posibles
                all_paths = self._explore_all_paths(
                    sensor, set_id, set_raised, reference_sensor, reference_set
                )
                
                if not all_paths:
                    # No hay caminos válidos
                    results.append(self._create_multipath_result_record(sensor, set_id, 
                                                                        status='Sin caminos válidos'))
                    continue
                
                # Extraer offsets y errores
                offsets = np.array([p[0] for p in all_paths])
                errors = np.array([p[1] for p in all_paths])
                
                # 1. Primer camino (estrategia actual)
                primer_offset = offsets[0]
                primer_error = errors[0]
                
                # 2. Camino de error mínimo
                idx_min = np.argmin(errors)
                min_error_offset = offsets[idx_min]
                min_error = errors[idx_min]
                
                # 3. Media ponderada por 1/error²
                # Pesos: w_i = 1/σ_i²
                weights_unnorm = 1.0 / (errors**2 + 1e-10)  # +epsilon para evitar división por cero
                # Media ponderada: μ = Σ(x_i × w_i) / Σ(w_i)
                weighted_offset = np.sum(offsets * weights_unnorm) / np.sum(weights_unnorm)
                # Error de la media ponderada: σ = 1/√(Σ(w_i)) = 1/√(Σ(1/σ_i²))
                weighted_error = 1.0 / np.sqrt(np.sum(weights_unnorm))
                
                # Estadísticas de variabilidad
                std_between_paths = np.std(offsets)
                max_diff = np.max(offsets) - np.min(offsets)
                
                results.append(self._create_multipath_result_record(
                    sensor, set_id, len(all_paths),
                    primer_offset, primer_error,
                    min_error_offset, min_error,
                    weighted_offset, weighted_error,
                    std_between_paths, max_diff,
                    'Calculado'
                ))
        
        df = pd.DataFrame(results).sort_values(['Set', 'Sensor'])
        
        # Imprimir resumen básico usando helper
        self._print_calibration_summary(df, "=== RESULTADOS MULTI-CAMINO ===")
        
        # Estadísticas específicas de multi-camino
        calculated = df[df['Status'] == 'Calculado']
        
        if len(calculated) > 0:
            print(f"Caminos promedio por sensor: {calculated['N_Caminos'].mean():.1f}")
            print(f"Sensores con múltiples caminos: {(calculated['N_Caminos'] > 1).sum()}")
            
            print("\n=== Diferencias entre estrategias ===")
            diff_primer_vs_minErr = np.abs(calculated['Constante_Primer_Camino_K'] - calculated['Constante_Min_Error_K'])
            diff_primer_vs_media = np.abs(calculated['Constante_Primer_Camino_K'] - calculated['Constante_Media_Ponderada_K'])
            
            print("Primer camino vs Min Error:")
            print(f"  Diferencia promedio: {diff_primer_vs_minErr.mean():.6f} K")
            print(f"  Diferencia máxima: {diff_primer_vs_minErr.max():.6f} K")
            
            print("\nPrimer camino vs Media Ponderada:")
            print(f"  Diferencia promedio: {diff_primer_vs_media.mean():.6f} K")
            print(f"  Diferencia máxima: {diff_primer_vs_media.max():.6f} K")
            
            print("\n=== Variabilidad entre caminos ===")
            print(f"Std promedio entre caminos: {calculated['Std_Entre_Caminos_K'].mean():.6f} K")
            print(f"Max diferencia promedio: {calculated['Max_Diff_Caminos_K'].mean():.6f} K")
            
            # Sensores con alta variabilidad (posibles problemas)
            high_var = calculated[calculated['Std_Entre_Caminos_K'] > 0.01]
            if len(high_var) > 0:
                print(f"\n⚠ {len(high_var)} sensores con alta variabilidad (std > 0.01 K)")
        
        return df
    
    def _get_reference_set(self) -> float:
        """Obtiene el set de referencia (mayor ronda, mayor número)"""
        max_round = max(self.sets_by_round.keys())
        sets_in_max_round = self.sets_by_round[max_round]
        
        if not sets_in_max_round:
            # Si no hay sets en la ronda máxima, buscar en la anterior
            max_round -= 1
            sets_in_max_round = self.sets_by_round[max_round]
        
        return max(sets_in_max_round)
    
    def _get_all_sensors(self) -> List[int]:
        """Obtiene lista de todos los sensores en todos los sets"""
        all_sensors = set()
        sets_config = self.config.get('sensors', {}).get('sets', {})
        
        for set_id in self.sets.keys():
            set_config = sets_config.get(float(set_id), {})
            sensors = set_config.get('sensors', [])
            all_sensors.update(sensors)
        
        return sorted(list(all_sensors))
    
    def get_summary(self) -> pd.DataFrame:
        """
        Genera un resumen de la estructura del árbol.
        
        Returns:
            DataFrame con información de cada set
        """
        summary = []
        sets_config = self.config.get('sensors', {}).get('sets', {})
        
        for set_id in sorted(self.sets.keys()):
            set_config = sets_config.get(float(set_id), {})
            sensors = set_config.get('sensors', [])
            raised = set_config.get('raised', [])
            round_num = set_config.get('round', 1)
            
            summary.append({
                'Set': set_id,
                'Round': round_num,
                'N_Sensors': len(sensors),
                'N_Raised': len(raised),
                'Raised_Sensors': raised if raised else []
            })
        
        df = pd.DataFrame(summary)
        return df
    
    def print_tree_structure(self):
        """Imprime la estructura del árbol de forma legible"""
        print("\n" + "="*70)
        print("ESTRUCTURA DEL ARBOL DE CALIBRACION")
        print("="*70)
        
        sets_config = self.config.get('sensors', {}).get('sets', {})
        
        for round_num in sorted(self.sets_by_round.keys(), reverse=True):
            sets = self.sets_by_round[round_num]
            if not sets:
                continue
            
            print(f"\nRONDA {round_num} ({len(sets)} sets)")
            print("-" * 70)
            
            for set_id in sets:
                set_config = sets_config.get(float(set_id), {})
                sensors = set_config.get('sensors', [])
                raised = set_config.get('raised', [])
                
                print(f"  Set {int(set_id):2d}: {len(sensors):2d} sensores", end="")
                if raised:
                    print(f" | {len(raised)} raised: {raised}")
                else:
                    print()
        
        print("="*70)
    
    def export_all_paths_details(self, reference_set: Optional[float] = None,
                                  r1_sets_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        Exporta TODOS los caminos individuales para cada sensor (similar a calibration_details_tree.csv).
        
        Cada fila representa UN camino completo con sus pasos intermedios.
        Un sensor con N caminos posibles tendrá N filas en el CSV.
        
        Args:
            reference_set: ID del set de referencia R3
            r1_sets_range: Tupla (min, max) para limitar sets R1
        
        Returns:
            DataFrame con todos los caminos individuales detallados
        """
        if reference_set is None:
            reference_set = self._get_reference_set()
        
        print("\n=== EXPORTANDO TODOS LOS CAMINOS INDIVIDUALES ===")
        
        # Determinar sets R1
        r1_sets = self._determine_r1_sets(reference_set, r1_sets_range)
        
        # Obtener referencia
        reference_sensor, _ = self._get_reference_info(reference_set)
        if reference_sensor is None:
            return pd.DataFrame()
        
        print(f"Referencia absoluta: Sensor {reference_sensor} (Set {reference_set})")
        print(f"Procesando {len(r1_sets)} sets de Ronda 1\n")
        
        all_paths_records = []
        
        for set_id in sorted(r1_sets):
            sets_config = self.config.get('sensors', {}).get('sets', {})
            set_config = sets_config.get(float(set_id), {})
            set_sensors = set_config.get('sensors', [])
            set_raised = set_config.get('raised', [])
            set_discarded = set_config.get('discarded', [])
            
            if not set_raised:
                continue
            
            for sensor in set_sensors:
                if sensor in set_discarded:
                    continue
                
                # Caso especial: sensores raised
                if sensor in set_raised:
                    other_raised = [r for r in set_raised if r != sensor]
                    if not other_raised:
                        continue
                    
                    ref_raised = other_raised[0]
                    offset_0, error_0 = self.get_offset_within_set(set_id, sensor, ref_raised)
                    
                    if offset_0 is None:
                        continue
                    
                    if error_0 is None:
                        error_0 = 0.0
                    
                    # Explorar caminos para ref_raised
                    paths_ref = self._explore_all_paths(
                        ref_raised, set_id, set_raised, reference_sensor, reference_set
                    )
                    
                    if not paths_ref:
                        continue
                    
                    # Aplicar offset_0 a todos los caminos
                    for path_idx, (offset_ref, error_ref, path_info) in enumerate(paths_ref, 1):
                        total_offset = offset_0 + offset_ref
                        total_error = propagate_error(error_0, error_ref)
                        
                        # Reconstruir TODOS los pasos desde path_info
                        # path_info contiene el camino COMPLETO desde ref_raised:
                        #   offset_1: ref_raised → raised_r1 (dentro de R2)
                        #   offset_2: raised_r1 → bridge_r2_r3 (dentro de R2)  
                        #   offset_3: bridge_r2_r3 → reference (en R3)
                        
                        all_paths_records.append({
                            'Sensor': sensor,
                            'Set': set_id,
                            'Round': 1,
                            'Path_Number': path_idx,
                            
                            # Paso 0: sensor_raised → otro_raised (dentro del mismo set R1)
                            'Paso0_From': sensor,
                            'Paso0_To': ref_raised,
                            'Paso0_Set': set_id,
                            'Paso0_Offset_K': offset_0,
                            'Paso0_Error_K': error_0,
                            
                            # Paso 1: ref_raised → raised_r1 (en R2)
                            'Paso1_From': ref_raised,
                            'Paso1_To': path_info['raised_r1'],
                            'Paso1_Set': path_info['set_r2'],
                            'Paso1_Offset_K': path_info['offset_1'],
                            'Paso1_Error_K': path_info['error_1'],
                            
                            # Paso 2: raised_r1 → bridge_r2_r3 (en R2)
                            'Paso2_From': path_info['raised_r1'],
                            'Paso2_To': path_info['bridge_r2_r3'],
                            'Paso2_Set': path_info['set_r2'],
                            'Paso2_Offset_K': path_info['offset_2'],
                            'Paso2_Error_K': path_info['error_2'],
                            
                            # Paso 3: bridge_r2_r3 → reference (en R3)
                            'Paso3_From': path_info['bridge_r2_r3'],
                            'Paso3_To': reference_sensor,
                            'Paso3_Set': reference_set,
                            'Paso3_Offset_K': path_info['offset_3'],
                            'Paso3_Error_K': path_info['error_3'],
                            'Paso3_From': None,
                            'Paso3_Offset_K': path_info['offset_3'],
                            'Paso3_Error_K': path_info['error_3'],
                            
                            # Total
                            'Total_Offset_K': total_offset,
                            'Total_Error_K': total_error
                        })
                    continue
                
                # Sensores normales (no raised)
                all_paths = self._explore_all_paths(
                    sensor, set_id, set_raised, reference_sensor, reference_set
                )
                
                if not all_paths:
                    continue
                
                # Exportar cada camino individualmente
                for path_idx, (total_offset, total_error, path_info) in enumerate(all_paths, 1):
                    all_paths_records.append({
                        'Sensor': sensor,
                        'Set': set_id,
                        'Round': 1,
                        'Path_Number': path_idx,
                        
                        # Paso 1: sensor → raised_r1
                        'Paso1_From': sensor,
                        'Paso1_To': path_info['raised_r1'],
                        'Paso1_Set': set_id,
                        'Paso1_Offset_K': path_info['offset_1'],
                        'Paso1_Error_K': path_info['error_1'],
                        
                        # Paso 2: raised_r1 → bridge_r2_r3
                        'Paso2_From': path_info['raised_r1'],
                        'Paso2_To': path_info['bridge_r2_r3'],
                        'Paso2_Set': path_info['set_r2'],
                        'Paso2_Offset_K': path_info['offset_2'],
                        'Paso2_Error_K': path_info['error_2'],
                        
                        # Paso 3: bridge_r2_r3 → reference
                        'Paso3_From': path_info['bridge_r2_r3'],
                        'Paso3_To': reference_sensor,
                        'Paso3_Set': reference_set,
                        'Paso3_Offset_K': path_info['offset_3'],
                        'Paso3_Error_K': path_info['error_3'],
                        
                        # Total
                        'Total_Offset_K': total_offset,
                        'Total_Error_K': total_error
                    })
        
        df = pd.DataFrame(all_paths_records)
        
        if len(df) > 0:
            # Ordenar por Sensor, Set, Path_Number
            df = df.sort_values(['Sensor', 'Set', 'Path_Number']).reset_index(drop=True)
            
            print(f"\n✓ Total caminos exportados: {len(df)}")
            print(f"✓ Total sensores únicos: {df['Sensor'].nunique()}")
            print(f"✓ Caminos promedio por sensor: {len(df) / df['Sensor'].nunique():.1f}")
        else:
            print("\n⚠️ No se encontraron caminos para exportar")
        
        return df
