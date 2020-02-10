import requests
import time

start_time = time.time()
for i in range(0, 10):
    images = []
    with open(r"0166.png", 'rb') as f:
        image_bytes = f.read()

    res = requests.post('http://10.1.56.30:15555/vision/supplier/person/batch_mask' , files = {'file' : image_bytes} , stream = True)   
    
    if res.content is not None:
        save_image_name = "save-" + str(i) + ".jpg" 
        with open(save_image_name , 'wb') as w:
            w.write(res.content)


print('cost time: ' , time.time() - start_time)
