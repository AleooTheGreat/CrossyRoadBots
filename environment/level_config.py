
class LevelConfig:
    
    def __init__(self, name, min_speed, max_speed, min_vehicles, max_vehicles, 
                 truck_frequency, safe_zone_spacing):
        self.name = name
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles
        self.truck_frequency = truck_frequency
        self.safe_zone_spacing = safe_zone_spacing
    
    def __repr__(self):
        return f"LevelConfig(name='{self.name}')"


class LevelManager:
    
    LEVELS = {
        'easy': LevelConfig(
            name='Easy',
            min_speed=0.03,
            max_speed=0.08,
            min_vehicles=1,
            max_vehicles=2,
            truck_frequency=15,  
            safe_zone_spacing=4 
        ),
        
        'medium': LevelConfig(
            name='Medium',
            min_speed=0.06,
            max_speed=0.12,
            min_vehicles=2,
            max_vehicles=4,
            truck_frequency=25,
            safe_zone_spacing=5
        ),
        
        'medium-hard': LevelConfig(
            name='Medium-Hard',
            min_speed=0.10,
            max_speed=0.14,
            min_vehicles=3,
            max_vehicles=5,
            truck_frequency=35, 
            safe_zone_spacing=6 
        )
    }
    
    @classmethod
    def get_level(cls, level_name='medium'):
        level_name = level_name.lower()
        if level_name not in cls.LEVELS:
            print(f"Warning: Level '{level_name}' not found.")
            level_name = 'medium'
        return cls.LEVELS[level_name]
    
    @classmethod
    def get_all_levels(cls):
        return list(cls.LEVELS.keys())
    
    @classmethod
    def create_custom_level(cls, name, min_speed, max_speed, min_vehicles, 
                           max_vehicles, truck_frequency, safe_zone_spacing):
        return LevelConfig(name, min_speed, max_speed, min_vehicles, 
                          max_vehicles, truck_frequency, safe_zone_spacing)
