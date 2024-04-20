This directory has a compose file you can use to serve your model directly using tensorflow serve, and then perform 
inference using the REST api (you can also use the gRPC one if you want)

To perform inference, first start the container:
```
docker-compose up
```
> **_NOTE:_**  Since Im using a Mac, i had to use the bitnami image, if you want the regular tensorflow serve image from tensorflow just switch the image out in the compose file.


Next, hit the REST api using curl like below (you can also do this using a client)
```
curl -X POST http://localhost:8501/v1/models/iotdnn:predict \
     -H "Content-Type: application/json" \
     -d '{
           "instances": [
             [0.0,54.0,6.0,64.0,0.3298071530725829,0.3298071530725829,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,567.0,54.0,54.0,54.0,0.0,54.0,83343831.92013878,9.5,10.392304845413264,0.0,0.0,0.0,141.55]
           ]
         }'
```

you should get a reply like:
```
{
    "predictions": [[0.176688135, 0.00948377512, 0.251042336, 0.145553797, 0.118325919, 0.154552132, 0.123684727, 0.0206692163]
    ]
}
```
which are the probabilities associated with each attack type, the mappings are:
```
['Benign' 'BruteForce' 'DDoS' 'DoS' 'Mirai' 'Recon' 'Spoofing' 'Web']
```