import requests
import time

start_time = time.time()
for i in range(0, 100):
    with open(r"cat.jpg", 'rb') as f:
        image_bytes = f.read()

    res = requests.post('http://10.1.41.46:5005/predict' , files = {'file' : image_bytes} , stream = True)   
    print(res.text)

print('cost time: ' , time.time() - start_time)
