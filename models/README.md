## Install project ##
Create a virtualenv named `erica_prediction_env`, making sure it uses python2.7, activate it and install the requirements in it (the path to python2.7 might be a different one on your system):
```
virtualenv -p /usr/bin/python2.7 erica_prediction_env
source erica_prediction_env/bin/activate
pip install -r requirements.txt
```

## Train and export a model ##
TODO

## Upload a model to google cloud ##
Open the google cloud console in a browser. Upload the exported folder, named something like `my_export_dir/1498119228618/`, to a bucket in google cloud storage. Then go to ML engine, create a model, then a version inside it, specifying the uploaded folder as source.
