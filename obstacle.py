import numpy as np
import random
import matplotlib.pyplot as plt
import math




def perpendicular_distance(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    numerator = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    distance = numerator / denominator
    return distance






def path_check(path, obstacles, distance_factor):
    for i in range(len(path) - 1):
        point1 = path[i]
        point2 = path[i + 1]
        
        for obstacle in obstacles:
            dist = perpendicular_distance(obstacle,point1, point2)
            if dist < distance_factor:
                return False  
    return True 
    

    
    
    

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def aco1(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q,obstacles):
    points=np.array(points)
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point]**alpha / distance(points[current_point], points[unvisited_point])**beta
                
                probabilities /= np.sum(probabilities)
                
                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point
            
            paths.append(path)
            path_lengths.append(path_length)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        
        pheromone *= evaporation_rate
        
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(points[:,0], points[:,1] ,c='r', marker='o')
    
    
    
    
    for i in range(n_points-1):
        ax.plot([points[best_path[i],0], points[best_path[i+1],0]],
                [points[best_path[i],1], points[best_path[i+1],1]],
                c='g', linestyle='-', linewidth=2, marker='o')
        
    ax.plot([points[best_path[0],0], points[best_path[-1],0]],
            [points[best_path[0],1], points[best_path[-1],1]],
            c='g', linestyle='-', linewidth=2, marker='o')
    for obstacle in obstacles:
        ax.scatter(obstacle[0], obstacle[1], c='b', marker='s', s=10)
   
    for obstacle in obstacles:
        ax.scatter(obstacle[0], obstacle[1], c='b', marker='s', s=10)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    plt.show()


# part-3
# obstacles = np.random.uniform(0, 500, size=(105, 2))                    

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def aco(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q,obstacles,distance_factor):
    # print(type(points))
    points=np.array(points)
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    all_paths=[]
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point]**alpha / distance(points[current_point], points[unvisited_point])**beta
                
                probabilities /= np.sum(probabilities)
                
                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point
            
            paths.append(path)
            path_lengths.append(path_length)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        
        pheromone *= evaporation_rate
        all_paths.extend(list(zip(paths, path_lengths)))
        all_paths.sort(key=lambda x: x[1])
               
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(points[:,0], points[:,1] ,c='r', marker='o')
    
    
    
    
    for i in range(n_points-1):
        ax.plot([points[best_path[i],0], points[best_path[i+1],0]],
                [points[best_path[i],1], points[best_path[i+1],1]],
                c='g', linestyle='-', linewidth=2, marker='o')
        
    ax.plot([points[best_path[0],0], points[best_path[-1],0]],
            [points[best_path[0],1], points[best_path[-1],1]],
            c='g', linestyle='-', linewidth=2, marker='o')
    for obstacle in obstacles:
        ax.scatter(obstacle[0], obstacle[1], c='b', marker='s', s=10)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    print(best_path)
    print(best_path_length)
    plt.show()
    
    

#    
    
                        
                
                
                    
                   
                
            
            
        
        
        
def perpendicular_distance(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    print(point)
    print(line_start)
    print(line_end)
    numerator = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    distance = numerator / denominator
    return distance


def point_on_line(obstacle,distance_factor, theta):
    x1, y1 = obstacle
    x = x1 -distance_factor * math.cos(theta)
    y = y1 - distance_factor * math.sin(theta)
    return [x, y]



def angle_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the angle in radians using arctangent
    theta = math.atan2(y2 - y1, x2 - x1)

    return theta



def perpendicular_point_on_line(point_a, point_b, obstacle_c):
    vector_ab = point_b - point_a
    vector_ac = obstacle_c - point_a
    scalar_projection = np.dot(vector_ac, vector_ab) / np.dot(vector_ab, vector_ab)
    point_d = point_a + scalar_projection * vector_ab
    
    return point_d

def path_check1(path, obstacles, distance_factor):
    sol=[]
    j=0
    for i in range(len(path)-1+j):
        point1 = path[i]
        point2 = path[i + 1]
        print(distance_factor)
        sol.append(point1)
        for obstacle in obstacles:
            dist = distance(obstacle,(point1+point2)/2)
            if dist < distance_factor:
                print(dist)
                ppoint = (point1+point2)/2
                theta=angle_between_points(ppoint,obstacle)
                new_p=point_on_line(obstacle,distance_factor,theta)
                new_p=np.array(new_p)
                sol.append(new_p)
                j=j+1
    
    point1 = path[len(path)-1]
    point2 = path[0] 
    ppoint = (point1+point2)/2 
    theta=angle_between_points(ppoint,obstacle)
    new_p=point_on_line(obstacle,distance_factor,theta)
    new_p=np.array(new_p)
    sol.append(new_p)    
    sol.append(path[len(path)-1])  
            
    return sol       
        
        
        
        
def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q,distance_factor,obstacles):
    print(type(points))
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point]**alpha / distance(points[current_point], points[unvisited_point])**beta
                
                probabilities /= np.sum(probabilities)
                
                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point
            
            paths.append(path)
            path_lengths.append(path_length)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
             
        pheromone *= evaporation_rate
        
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length
    
    fig = plt.figure(figsize=(8, 6))
    ax =  fig.add_subplot(111)
    # ax.scatter(points[:,0], points[:,1], c='r', marker='o')
    path_coordinates = [points[idx] for idx in best_path]
    print(path_coordinates)
    best_path=path_check1(path_coordinates,obstacles,distance_factor)
    print(best_path)       
    # for i in range(n_points-1):
    #     ax.plot([best_path[i][0], best_path[i+1][0]],
    #             [best_path[i][1], best_path[i+1][1]],
    #             c='g', linestyle='-', linewidth=2, marker='o')
        
    # ax.plot([best_path[0][0], best_path[len(best_path)-1][0]],
    #             [best_path[0][1], best_path[len(best_path)-1][1]],
    #         c='r', linestyle='-', linewidth=2, marker='o')   
    # ax.set_xlim([-10, 10])
    # ax.set_ylim([-10, 10])
    
    # for obstacle in obstacles:
    #     ax.scatter(obstacle[0], obstacle[1], c='b', marker='s', s=10)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # plt.savefig("graph{}".format(n_ants))
    # plt.show()    
    aco1(points, n_ants=11, n_iterations=50, alpha=1, beta=3, evaporation_rate=0.5, Q=1,obstacles=obstacles)   
    aco(best_path, n_ants=11, n_iterations=50, alpha=1, beta=3, evaporation_rate=0.5, Q=1,obstacles=obstacles,distance_factor=distance_factor)




data=[[1,2],[1,4],[4,2],[4,4]]
obstacles =[[2.5,3]]     
obstacles=np.array(obstacles)
data_array = np.array(data)
                        
points=data_array
ant_colony_optimization(points, n_ants=11, n_iterations=50, alpha=1, beta=1, evaporation_rate=0.5, Q=1,distance_factor=2,obstacles=obstacles)        
        
       

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 