import numpy as np
import datetime

start_time = datetime.datetime.now()

vector1 = np.array([1, 1])
vector2 = np.array([2, 2])
print(np.linalg.norm(vector1-vector2))

end_time = datetime.datetime.now()
print('time_taken = ', (end_time - start_time).seconds)
