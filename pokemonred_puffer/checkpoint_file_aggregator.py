import os
from collections import defaultdict
from pokegym import game_map
import numpy as np

LOGGABLE_LOCATIONS = {
    "Viridian City": (1, None),
    "Viridian Forest Entrance": (51, None),
    "Viridian Forest Exit": (47, None),
    "Pewter City": (2, None),
    "Badge 1": (1001, None),
    "Route 3": (14, None),
    "Mt Moon Entrance (Route 3)": (59, None),
    "Mt Moon B1F": (60, None),
    "Mt Moon B2F": (61, None),
    "Mt Moon Exit": (59, None),
    "Route 4": (15, None),
    "Cerulean City": (3, None),
    "Badge 2": (1002, None),
    "Bill": (88, None),
    "Vermilion City": (5, None),
    "Vermilion Harbor": (94, None),
    "SS Anne Start": (95, None),
    "SS Anne Captains Office": (101, None)
}

def read_checkpoint_logs():
    checkpoint_data = defaultdict(str)
    stats_data = defaultdict(lambda: {'mean': 0, 'variance': 0, 'std_dev': 0, 'environments_percents': 0})

    base_dir = "/puffertank/thatguy/pokemonred_puffer/experiments/"
    file_path = "/puffertank/thatguy/pokemonred_puffer/experiments/running_experiment.txt"
    
    with open(file_path, "r") as pathfile:
        exp_uuid8 = pathfile.readline().strip()

    sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
    
    os.makedirs(sessions_path, exist_ok=True)
    all_location_times = defaultdict(list)
    current_location_times = defaultdict(lambda: float('inf'))

    try:
        for folder in os.listdir(sessions_path):
            session_path = os.path.join(sessions_path, folder)

            checkpoint_log_file = None
            for file in os.listdir(session_path):
                if file.startswith("checkpoint_log") and file.endswith(".txt"):
                    checkpoint_log_file = os.path.join(session_path, file)
                    break

            if checkpoint_log_file is not None:
                with open(checkpoint_log_file, 'r') as log_file:
                    for line in log_file:
                        line = line.strip()
                        if line.startswith("Location ID:"):
                            current_location_id = line.split(":")[-1].strip()
                            if current_location_id.isdigit():
                                current_location_id = int(current_location_id)
                                if current_location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
                                    current_location = current_location_id
                                else:
                                    current_location = 40
                            else:
                                current_location = 40
                        elif line.startswith("Time Visited:") and current_location is not None:
                            time_visited = float(line.split(":")[-1].strip())
                            current_location_times[current_location] = min(current_location_times[current_location], time_visited)
                            all_location_times[current_location].append(time_visited)

        total_environments = len(os.listdir(sessions_path))

        for location_id, time_visited in current_location_times.items():
            location_name = game_map.get_map_name_from_map_n(location_id)
            formatted_time = '{:.2f}'.format(time_visited / 60)
            checkpoint_data[location_name] = formatted_time
            environments_reached = len(all_location_times[location_id])
            environments_percent = min((environments_reached / total_environments) * 100, 100)  # Cap percentage at 100%
            stats_data[location_name]['environments_percents'] = round(environments_percent, 2)


        # for location_id, time_visited in current_location_times.items():
        #     location_name = game_map.get_map_name_from_map_n(location_id)
        #     formatted_time = '{:.2f}'.format(time_visited / 60)
        #     checkpoint_data[location_name] = formatted_time
        #     environments_reached = len(all_location_times[location_id])
        #     environments_percent = (environments_reached / total_environments) * 100
        #     stats_data[location_name]['environments_percents'] = environments_percent

        for location_id, times in all_location_times.items():
            if times:
                location_name = game_map.get_map_name_from_map_n(location_id)
                mean = np.mean(times) / 60  
                variance = np.var(times) / 3600  
                std_dev = np.sqrt(variance)  
                stats_data[location_name]['mean'] = '{:.2f}'.format(mean)
                stats_data[location_name]['variance'] = '{:.2f}'.format(variance)
                stats_data[location_name]['std_dev'] = '{:.2f}'.format(std_dev)

        return checkpoint_data, stats_data

    except Exception as e:
        print("An error occurred:", e)
        viridian_city_id = LOGGABLE_LOCATIONS['Viridian City'][0]
        checkpoint_data[game_map.get_map_name_from_map_n(viridian_city_id)] = '0.00'
        
        stats_data[game_map.get_map_name_from_map_n(viridian_city_id)] = {
            'mean': '0.00',
            'variance': '0.00',
            'std_dev': '0.00',
            'environments_percents': 0
        }
        return checkpoint_data, stats_data

checkpoint_dict, stats_dict = read_checkpoint_logs()
if checkpoint_dict is not None and stats_dict is not None:
    with open('checkpoint_dict.txt', 'w') as f:
        f.write(str(checkpoint_dict))
        f.write('\n')
        f.write(str(stats_dict))


















# import os
# from collections import defaultdict
# from pokegym import game_map
# import pdb as T
# import numpy as np

# # from memory_profiler import profile

# LOGGABLE_LOCATIONS = {"Viridian City": (1, None),
#     "Viridian Forest Entrance": (51, None),
#     "Viridian Forest Exit": (47, None),
#     "Pewter City": (2, None),
#     "Badge 1": (None, None),
#     "Route 3": (14, None),
#     "Mt Moon Entrance (Route 3)": (59, None),
#     "Mt Moon B1F": (60, None),
#     "Mt Moon B2F": (61, None),
#     "Mt Moon Exit": (59, None),
#     "Route 4": (15, None),
#     "Cerulean City": (3, None),
#     "Badge 2": (None, None),
#     "Bill": (88, None),
#     "Vermilion City": (5, None),
#     "Vermilion Harbor": (94, None),
#     "SS Anne Start": (95, None),
#     "SS Anne Captains Office": (101, None)}

# # @profile
# def read_checkpoint_logs():
#     checkpoint_data = defaultdict(str)
#     stats_data = defaultdict(lambda: {'mean': 0, 'variance': 0, 'std_dev': 0})

#     base_dir = "/puffertank/thatguy/pokemonred_puffer/experiments/"
#     file_path = "/puffertank/thatguy/pokemonred_puffer/experiments/running_experiment.txt"
    
#     print(f'base_dir={base_dir}')
#     print(f'file_path={file_path}')

#     with open(file_path, "r") as pathfile:
#         exp_uuid8 = pathfile.readline().strip()

#     sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
#     print(f'sessions_path={sessions_path}')
    
#     os.makedirs(sessions_path, exist_ok=True)
#     all_location_times = defaultdict(list)
#     current_location_times = defaultdict(lambda: float('inf'))

#     try:
#         # Iterate over each log file and append times to all_location_times
#         for folder in os.listdir(sessions_path):
#             print(f'sessions_path={sessions_path}')
#             print(f'folder={folder}')
#             session_path = os.path.join(sessions_path, folder)

#             checkpoint_log_file = None
#             for file in os.listdir(session_path):
#                 if file.startswith("checkpoint_log") and file.endswith(".txt"):
#                     checkpoint_log_file = os.path.join(session_path, file)
#                     break

#             if checkpoint_log_file is not None:
#                 with open(checkpoint_log_file, 'r') as log_file:
#                     # lines = log_file.readlines()

#                     for line in log_file:
#                         line = line.strip()
#                         if line.startswith("Location ID:"):
#                             current_location_id = line.split(":")[-1].strip()
#                             if current_location_id.isdigit():
#                                 current_location_id = int(current_location_id)
#                                 if current_location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#                                     current_location = current_location_id
#                                 else:
#                                     current_location = 40
#                             else:
#                                 current_location = 40
#                         elif line.startswith("Time Visited:") and current_location is not None:
#                             time_visited = float(line.split(":")[-1].strip())
#                             current_location_times[current_location] = min(current_location_times[current_location], time_visited)
#                             all_location_times[current_location].append(time_visited)
#                             # print(f'all loc: {all_location_times}')

#         # Update checkpoint_data with minimum times
#         for location_id, time_visited in current_location_times.items():
#             location_name = game_map.get_map_name_from_map_n(location_id)
#             formatted_time = '{:.2f}'.format(time_visited / 60)
#             checkpoint_data[location_name] = formatted_time

#         # Calculate mean, variance, and standard deviation for each location
#         for location_id, times in all_location_times.items():
#             if times:
#                 location_name = game_map.get_map_name_from_map_n(location_id)
                
#                 mean = np.mean(times) / 60  # Convert mean to minutes
#                 variance = np.var(times) / 3600  # Convert variance to minutes^2
#                 std_dev = np.sqrt(variance)  # Standard deviation in minutes
#                 stats_data[location_name]['mean'] = '{:.2f}'.format(mean)
#                 stats_data[location_name]['variance'] = '{:.2f}'.format(variance)
#                 stats_data[location_name]['std_dev'] = '{:.2f}'.format(std_dev)



#         # # Calculate mean, variance, and standard deviation for each location
#         # for location_id, times in all_location_times.items():
#         #     if times:
#         #         location_name = game_map.get_map_name_from_map_n(location_id)
                
#         #         mean = np.mean(times)
#         #         variance = np.var(times)
#         #         std_dev = np.sqrt(variance)
#         #         stats_data[location_name]['mean'] = '{:.2f}'.format(mean)
#         #         stats_data[location_name]['variance'] = '{:.2f}'.format(variance)
#         #         stats_data[location_name]['std_dev'] = '{:.2f}'.format(std_dev)

#         return checkpoint_data, stats_data

#     except Exception as e:
#         print("An error occurred:", e)
#         # Set default values for 'Viridian City'
#         viridian_city_id = LOGGABLE_LOCATIONS['Viridian City'][0]
#         checkpoint_data[game_map.get_map_name_from_map_n(viridian_city_id)] = '0.00'
        
#         stats_data[game_map.get_map_name_from_map_n(viridian_city_id)] = {
#             'mean': '0.00',
#             'variance': '0.00',
#             'std_dev': '0.00'
#         }
#         return checkpoint_data, stats_data

# checkpoint_dict, stats_dict = read_checkpoint_logs()
# if checkpoint_dict is not None and stats_dict is not None:
#     with open('checkpoint_dict.txt', 'w') as f:
#         f.write(str(checkpoint_dict))
#         f.write('\n')
#         f.write(str(stats_dict))
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# if mean_sigma_dict is not None:
#     with open('mean_sigma_dict.txt', 'w') as f:
#         f.write(str(mean_sigma_dict))

# def read_checkpoint_logs():
#     checkpoint_data = defaultdict(str)
#     mean_sigma_data = defaultdict(lambda: {'mean': 0, 'sigma': 0})

#     base_dir = "experiments"
#     file_path = "experiments/running_experiment.txt"

#     with open(file_path, "r") as pathfile:
#         exp_uuid8 = pathfile.readline().strip()

#     sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
#     output_file_path = os.path.join(base_dir, exp_uuid8, "checkpoints_log.txt")

#     try:
#         for folder in os.listdir(sessions_path):
#             session_path = os.path.join(sessions_path, folder)
#             checkpoint_log_file = None
#             for file in os.listdir(session_path):
#                 if file.startswith("checkpoint_log") and file.endswith(".txt"):
#                     checkpoint_log_file = os.path.join(session_path, file)
#                     break
            
#             if checkpoint_log_file is not None:
#                 with open(checkpoint_log_file, 'r') as log_file:
#                     lines = log_file.readlines()
        
#                 current_location_times = defaultdict(lambda: float('inf'))
#                 current_location_fns = defaultdict(lambda: float('inf')) # added for mean and sigma calc

#                 for line in lines:
#                     line = line.strip()
#                     if line.startswith("Location ID:"):
#                         current_location_id = line.split(":")[-1].strip()
#                         if current_location_id.isdigit() and int(current_location_id) in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#                             current_location = int(current_location_id)
#                         else:
#                             current_location = None  # Skip if not a valid location ID
#                     elif line.startswith("Time Visited:") and current_location is not None:
#                         time_visited = float(line.split(":")[-1].strip())
#                         current_location_times[current_location] = min(current_location_times[current_location], time_visited)
#                         current_location_fns[current_location] = current_location_fns[current_location], time_visited # added for mean and sigma calc

#                 for location_id, time_visited in current_location_times.items():
#                     location_name = game_map.get_map_name_from_map_n(location_id)
#                     formatted_time = '{:.2f}'.format(time_visited)
                    
#                     if location_name in checkpoint_data:
#                         existing_time = float(checkpoint_data[location_name])
#                         if time_visited < existing_time:
#                             checkpoint_data[location_name] = formatted_time
#                     else:
#                         checkpoint_data[location_name] = formatted_time

#                 # Logic in this for loop (uncomment)
#                 # for location_id, time_visited in current_location_fns.items():
#                 #     location_name_fn = game_map.get_map_name_from_map_n(location_id)
#                 #     formatted_time = '{:.2f}'.format(time_visited)
                    
#                 #     if location_name_fn in checkpoint_data:
#                 #         existing_time = float(checkpoint_data[location_name_fn])
#                 #         if time_visited < existing_time:
#                 #             checkpoint_data[location_name_fn] = formatted_time
#                 #     else:
#                 #         checkpoint_data[location_name] = formatted_time

#                 # print("Final checkpoint data:", checkpoint_data)
#         return checkpoint_data

#     except Exception as e:
#         print("An error occurred:", e)
#         return None

# checkpoint_dict = read_checkpoint_logs()
# if checkpoint_dict is not None:
#     with open('checkpoint_dict.txt', 'w') as f:
#         f.write(str(checkpoint_dict))



# import os
# from collections import defaultdict
# from pokegym import game_map
# import pdb as T
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # from memory_profiler import profile

# LOGGABLE_LOCATIONS = {"Viridian City": (1, None),
#     "Viridian Forest Entrance": (51, None),
#     "Viridian Forest Exit": (47, None),
#     "Pewter City": (2, None),
#     "Pewter Gym": (54, None),
#     "Badge 1": (None, None),
#     "Route 3": (14, None),
#     "Mt Moon Entrance (Route 3)": (59, None),
#     "Mt Moon B1F": (60, None),
#     "Mt Moon B2F": (61, None),
#     "Mt Moon Exit": (59, None),
#     "Route 4": (15, None),
#     "Cerulean City": (3, None),
#     "Cerulean Gym": (65, None),
#     "Badge 2": (None, None),
#     "Bill's Lab": (88, None),
#     "Vermilion City": (5, None),
#     "Vermilion Harbor": (94, None),
#     "SS Anne Start": (95, None),
#     "SS Anne Captains Office": (101, None)}


# def get_exp_paths():
#         file_path = "/puffertank/thatguy/pokemonred_puffer/experiments/running_experiment.txt"
#         with open(file_path, "r") as pathfile:
#             exp_uuid8 = pathfile.readline().strip()
#             print(f'exp_uuid8={exp_uuid8}')
            
#         base_dir = "/puffertank/thatguy/pokemonred_puffer/experiments"
#         sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
#         return base_dir, file_path, exp_uuid8, sessions_path

# def read_checkpoint_logs():
#     checkpoint_data = defaultdict(str)
#     stats_data = defaultdict(lambda: {'mean': 0, 'variance': 0, 'std_dev': 0})

#     base_dir, file_path, exp_uuid8, sessions_path = get_exp_paths()
#     # base_dir = "/puffertank/thatguy/pokemonred_puffer/experiments"
#     # file_path = "/puffertank/thatguy/pokemonred_puffer/experiments/running_experiment.txt"

#     # with open(file_path, "r") as pathfile:
#     #     exp_uuid8 = pathfile.readline().strip()
#     #     print(f'exp_uuid8={exp_uuid8}')

#     # sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
#     # print(f'base_dir, file_path, exp_uuid8, sessions_path={base_dir, file_path, exp_uuid8, sessions_path}')
#     all_location_times = defaultdict(list)
#     current_location_times = defaultdict(lambda: float('inf'))
    

#     environment_counts = defaultdict(set)

#     for folder in os.listdir(sessions_path):
#         session_path = os.path.join(sessions_path, folder)
#         checkpoint_log_file = None
#         for file in os.listdir(session_path):
#             if file.startswith("checkpoint_log") and file.endswith(".txt"):
#                 checkpoint_log_file = os.path.join(session_path, file)
#                 break

#         if checkpoint_log_file is not None:
#             with open(checkpoint_log_file, 'r') as log_file:
#                 for line in log_file:
#                     line = line.strip()
#                     if line.startswith("Location ID:"):
#                         location_id = line.split(":")[-1].strip()
#                         # Assuming location_id is a valid identifier for the location
#                         # Add the folder (environment ID) to the set for this location
#                         if location_id.isdigit():
#                             environment_counts[int(location_id)].add(folder)

#             # Now log the count of unique environments per location
#             for location_id, environments in environment_counts.items():
#                 location_name = game_map.get_map_name_from_map_n(location_id)
#                 count = len(environments)
#                 logging.info(f"{location_name} (ID: {location_id}): {count} instances")


    
    
#     # Iterate over each log file and append times to all_location_times
#     for folder in os.listdir(sessions_path):
#         session_path = os.path.join(sessions_path, folder)
#         checkpoint_log_file = None
#         for file in os.listdir(session_path):
#             print(f'session_path={session_path}')
#             if file.startswith("checkpoint_log") and file.endswith(".txt"):
#                 checkpoint_log_file = os.path.join(session_path, file)
#                 break

#         if checkpoint_log_file is not None:
#             with open(checkpoint_log_file, 'r') as log_file:
#                 print(f'checkpoint_log_file={checkpoint_log_file}')
#                 # lines = log_file.readlines()

#                 for line in log_file:
#                     line = line.strip()
#                     if line.startswith("Location ID:"):
#                         current_location_id = line.split(":")[-1].strip()
#                         if current_location_id.isdigit():
#                             current_location_id = int(current_location_id)
#                             if current_location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#                                 current_location = current_location_id
#                                 print(f'current_location={current_location}, session_path_basename={os.path.basename(session_path)}')
#                                 f'env_{os.path.basename(session_path)}_locations_visited_dict'
#                                 {'location': f'{current_location}'}
#                             else:
#                                 current_location = 40
#                         else:
#                             current_location = 40
#                     elif line.startswith("Time Visited:") and current_location is not None:
#                         time_visited = float(line.split(":")[-1].strip())
#                         current_location_times[current_location] = min(current_location_times[current_location], time_visited)
#                         all_location_times[current_location].append(time_visited)
#                         # print(f'all loc: {all_location_times}')
                        
                            

#     try:
#         environment_counts = defaultdict(set)

#         for folder in os.listdir(sessions_path):
#             session_path = os.path.join(sessions_path, folder)
#             checkpoint_log_file = None  # Initialize to None for each folder
#             for file in os.listdir(session_path):
#                 if file.startswith("checkpoint_log") and file.endswith(".txt"):
#                     checkpoint_log_file = os.path.join(session_path, file)
#                     break  # Break after finding the first matching file
            
#             # Move the check for checkpoint_log_file's existence inside the loop
#             if checkpoint_log_file is not None:
#                 with open(checkpoint_log_file, 'r') as log_file:
#                     for line in log_file:
#                         line = line.strip()
#                         if line.startswith("Location ID:"):
#                             location_id = line.split(":")[-1].strip()
#                             if location_id.isdigit():
#                                 environment_counts[int(location_id)].add(folder)

            
#                     # Now log the count of unique environments per location
#             # for location_id, environments in environment_counts.items():
#             #     location_name = game_map.get_map_name_from_map_n(location_id)
#             #     count = len(environments)
#             #     logging.info(f"{location_name} (ID: {location_id}): {count} instances")

#         # environment_counts = defaultdict(set)

#         # for folder in os.listdir(sessions_path):
#         #     session_path = os.path.join(sessions_path, folder)
#         #     checkpoint_log_file = None
#         #     for file in os.listdir(session_path):
#         #         if file.startswith("checkpoint_log") and file.endswith(".txt"):
#         #             checkpoint_log_file = os.path.join(session_path, file)
#         #             break

#             # if checkpoint_log_file is not None:
#             #     with open(checkpoint_log_file, 'r') as log_file:
#             #         for line in log_file:
#             #             line = line.strip()
#             #             if line.startswith("Location ID:"):
#             #                 location_id = line.split(":")[-1].strip()
#             #                 # Assuming location_id is a valid identifier for the location
#             #                 # Add the folder (environment ID) to the set for this location
#             #                 if location_id.isdigit():
#             #                     environment_counts[int(location_id)].add(folder)

#         # # Now log the count of unique environments per location
#         # for location_id, environments in environment_counts.items():
#         #     location_name = game_map.get_map_name_from_map_n(location_id)
#         #     count = len(environments)
#         #     logging.info(f"{location_name} (ID: {location_id}): {count} instances")

        
        
#         # # Iterate over each log file and append times to all_location_times
#         # for folder in os.listdir(sessions_path):
#         #     session_path = os.path.join(sessions_path, folder)
#         #     checkpoint_log_file = None
#         #     for file in os.listdir(session_path):
#         #         print(f'session_path={session_path}')
#         #         if file.startswith("checkpoint_log") and file.endswith(".txt"):
#         #             checkpoint_log_file = os.path.join(session_path, file)
#         #             break

#         #     if checkpoint_log_file is not None:
#         #         with open(checkpoint_log_file, 'r') as log_file:
#         #             print(f'checkpoint_log_file={checkpoint_log_file}')
#         #             # lines = log_file.readlines()

#         #             for line in log_file:
#         #                 line = line.strip()
#         #                 if line.startswith("Location ID:"):
#         #                     current_location_id = line.split(":")[-1].strip()
#         #                     if current_location_id.isdigit():
#         #                         current_location_id = int(current_location_id)
#         #                         if current_location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#         #                             current_location = current_location_id
#         #                             print(f'current_location={current_location}, session_path_basename={os.path.basename(session_path)}')
#         #                             f'env_{os.path.basename(session_path)}_locations_visited_dict'
#         #                             {'location': f'{current_location}'}
#         #                         else:
#         #                             current_location = 40
#         #                     else:
#         #                         current_location = 40
#                         # elif line.startswith("Time Visited:") and current_location is not None:
#                         #     time_visited = float(line.split(":")[-1].strip())
#                         #     current_location_times[current_location] = min(current_location_times[current_location], time_visited)
#                         #     all_location_times[current_location].append(time_visited)
#                         #     # print(f'all loc: {all_location_times}')
                            
                            
#         # Dictionary to hold the visited locations for each environment
#         environments_locations_visited = defaultdict(set)

#         for folder in os.listdir(base_dir):
#             session_path = os.path.join(base_dir, folder)
#             checkpoint_log_file = os.path.join(session_path, f"checkpoint_log_{folder}.txt")
            
#             try:
#                 with open(checkpoint_log_file, 'r') as log_file:
#                     print(f'checkpoint_log_file={checkpoint_log_file}')
#                     for line in log_file:
#                         line = line.strip()
#                         if line.startswith("Location ID:"):
#                             current_location_id = line.split(":")[-1].strip()
#                             if current_location_id.isdigit():
#                                 current_location_id = int(current_location_id)
#                                 # Check if location is loggable
#                                 if current_location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#                                     # Add the location to the set of visited locations for this environment
#                                     environments_locations_visited[folder].add(current_location_id)
#                                     print(f'current_location={current_location_id}, session_path_basename={folder}')
#                                     raise
#                                 else:
#                                     current_location = 40
#                             else:
#                                 current_location = 40
#                         elif line.startswith("Time Visited:") and current_location is not None:
#                             time_visited = float(line.split(":")[-1].strip())
#                             current_location_times[current_location] = min(current_location_times[current_location], time_visited)
#                             all_location_times[current_location].append(time_visited)
#                             print(f'all loc times: {all_location_times}')
            
#                         # Update checkpoint_data with minimum times
#                         for location_id, time_visited in current_location_times.items():
#                             location_name = game_map.get_map_name_from_map_n(location_id)
#                             formatted_time = '{:.2f}'.format(time_visited / 60)
#                             checkpoint_data[location_name] = formatted_time

#                         # Calculate mean, variance, and standard deviation for each location
#                         for location_id, times in all_location_times.items():
#                             if times:
#                                 location_name = game_map.get_map_name_from_map_n(location_id)
                                
#                                 mean = np.mean(times) / 60  # Convert mean to minutes
#                                 variance = np.var(times) / 3600  # Convert variance to minutes^2
#                                 std_dev = np.sqrt(variance)  # Standard deviation in minutes
#                                 stats_data[location_name]['mean'] = '{:.2f}'.format(mean)
#                                 stats_data[location_name]['variance'] = '{:.2f}'.format(variance)
#                                 stats_data[location_name]['std_dev'] = '{:.2f}'.format(std_dev)

#                         total_environments = len(os.listdir(sessions_path))  # Total number of environments
#                         environment_counts = defaultdict(int)  # Dictionary to hold counts of environments reaching each location
#                         print(f"LINE193 Total Environments: {total_environments}")
                        
#                         logging.info(f"Total Environments: {total_environments}")
#                         for location_id, count in environment_counts.items():
#                             location_name = game_map.get_map_name_from_map_n(location_id)
#                             logging.info(f"{location_name}: Count before percentage calculation = {count}")

#                             # Optional: Log detailed environment IDs for each location if feasible
#                             # This might require modifying the counting logic to store environment IDs
                            
#                         # Your existing code to calculate and log percentages...

#                         for location_id, count in environment_counts.items():
#                             location_name = game_map.get_map_name_from_map_n(location_id)
#                             percentage = (count / total_environments) * 100
#                             logging.info(f"{location_name}: {percentage:.2f}% environments reached")

                    
#                         # Iterate through all_location_times to count environments for each location
#                         for location_id, times in all_location_times.items():
#                             environment_counts[location_id] = len(times)  # Count of environments that reached this location
                        
#                         # Calculate and add percentages to stats_data
#                         for location_id, count in environment_counts.items():
#                             location_name = game_map.get_map_name_from_map_n(location_id)
#                             percentage = (count / total_environments) * 100  # Calculate percentage
#                             # Add percentage to stats_data for the location
#                             stats_data[location_name]['environments_percentage'] = '{:.2f}%'.format(percentage)

#                         return checkpoint_data, stats_data, environment_counts
                        
#             except FileNotFoundError:
#                 print(f"File not found: {checkpoint_log_file}")



#     except Exception as e:
#         print("An error occurred:", e)
#         # Set default values for 'Viridian City'
#         viridian_city_id = LOGGABLE_LOCATIONS['Viridian City'][0]
#         checkpoint_data[game_map.get_map_name_from_map_n(viridian_city_id)] = '0.00'
        
#         stats_data[game_map.get_map_name_from_map_n(viridian_city_id)] = {
#             'mean': '0.00',
#             'variance': '0.00',
#             'std_dev': '0.00'
#         }
#         return checkpoint_data, stats_data, environment_counts
    
# # Assuming environment_counts is populated as described
# def print_location_sets(environment_counts):
#     base_dir, file_path, exp_uuid8, sessions_path = get_exp_paths()
#     total_environments = len(os.listdir(sessions_path))  # Total number of environments
#     for location_id, env_set in environment_counts.items():
#         location_name = game_map.get_map_name_from_map_n(location_id)

#         # Make sure env_set is a set
#         if isinstance(env_set, set):
#             if location_id in [map_id for location, (map_id, _) in LOGGABLE_LOCATIONS.items() if map_id is not None]:
#                 logging.info(f"Location '{location_name}' (ID: {location_id}) was visited by {len(env_set)}/{total_environments} ({len(env_set)/total_environments}%) environments: {sorted(env_set)}")
#             else:
#                 pass
#         else:
#             logging.error(f"Expected a set for location ID {location_id}, but got: {type(env_set)}")

# def log_checkpoint_data(checkpoint_dict, stats_dict, environment_counts, total_environments):
#     with open('checkpoint_dict.txt', 'w') as f:
#         for location, data in checkpoint_dict.items():
#             # Check if data is a dictionary and has the expected keys
#             if isinstance(data, dict) and 'time' in data and 'first_env' in data:
#                 f.write(f"{location}: First Time - {data['time']} by Environment {data['first_env']}\n")
#             else:
#                 # Handle cases where data is not structured as expected
#                 f.write(f"{location}: Data format unexpected. Actual data: {data}\n")
        
#         f.write("\nStatistics for Each Location:\n")
#         for location, stats in stats_dict.items():
#             f.write(f"{location}: Mean - {stats['mean']}, Variance - {stats['variance']}, Std Dev - {stats['std_dev']}\n")
        
#         f.write("\nEnvironment Counts and Percentages:\n")
#         for location, envs in environment_counts.items():
#             location_name = game_map.get_map_name_from_map_n(location)  # Adjust based on your game_map structure
#             percentage = (len(envs) / total_environments) * 100
#             f.write(f"{location_name} (ID: {location}): {len(envs)}/{total_environments} ({percentage:.2f}%) Environments: {sorted(envs)}\n")

# def main():
#     base_dir, file_path, exp_uuid8, sessions_path = get_exp_paths()  # Get sessions_path
#     checkpoint_dict, stats_dict, environment_counts = read_checkpoint_logs()
#     total_environments = len(os.listdir(sessions_path))  # Now sessions_path is defined

#     if checkpoint_dict and stats_dict and environment_counts:
#         print(f'checkpoint_dict={checkpoint_dict}\nstats_dict={stats_dict}\nenvironment_counts={environment_counts}\n')
#         print_location_sets(environment_counts)
#         log_checkpoint_data(checkpoint_dict, stats_dict, environment_counts, total_environments)  # Correctly pass total_environments

# if __name__ == "__main__":
#     main()





    
# # checkpoint_dict, stats_dict = read_checkpoint_logs()
# # if checkpoint_dict is not None and stats_dict is not None:
# #     with open('checkpoint_dict.txt', 'w') as f:
# #         f.write(str(checkpoint_dict))
# #         f.write('\n')
# #         f.write(str(stats_dict))
