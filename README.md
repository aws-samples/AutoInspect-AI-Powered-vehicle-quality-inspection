**Automated vehicle inspection using Whisper model and fuzzymatching**



**Prerequisites**

1. Any code editor installed in local(vs-code/pycharm) or can use AWS sagemaker in AWS Cloud.
2. clone the github repository into a local folder.



**Repository structure**

```
-greengrass-onnx-asr - This contains the main code. Inside this following is the structure
    -artifacts
        -audio - a sample audio file used in this experiment
        -model - quantized whisper tiny model
        -tokens- tokens used by model
        -inference.py- inferencing code for the model
    -recipes
        -onnx-asr.json- receipe for automatic speech recognition for IoT greengrass
        -onnxruntime.json- receipe for onnxruntime for IoT greengrass
    -LICENSE
    -README.md

```


**Build the ONNX Runtime and ONNX Whisper component components**

1. Modify the deployment recipe for onnxruntime and onnx-asr with updated s3 bucket
2. Go to the recipes folder and publish the components

```
aws greengrassv2 create-component-version --inline-recipe fileb://onnx-asr.json
aws greengrassv2 create-component-version --inline-recipe fileb://onnxruntime.json

```
**Deploy the component to a target device**


1. Go to console of IoT greengrass console and deploy the receipes to core device created earlier
2. Deploy the lambda-onnx as mentioned in the blog
3. In MQTT publish to the topic "defectlogger/trigger" with payload 
```
{"key":"sample_1.wav"}

```
4. Go to MQTT Topic and subscribe to topic "audioDevice/data"
5. Results can be seen in MQTT console.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

