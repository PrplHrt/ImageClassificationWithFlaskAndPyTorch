# Image Classification with Flask and PyTorch

This is a Flask web application that allows users to upload images and get predictions from a pre-trained PyTorch model. Currently, this application only works with images that are of the shape (3, h, w).

## Getting started

To get started, clone the repository to your local machine:

```
$ git clone https://github.com/PrplHrt/ImageClassificationWithFlaskAndPyTorch.git
$ cd ImageClassificationWithFlaskAndPyTorch
```

### Prerequisites

Before you can run the Flask application, you need to install the required dependencies. You can do this using `pip`:

```
$ pip install -r requirements.txt
```

### Running the application

To run the Flask application, you can use the following command:

```
$ export FLASK_APP=app.py
$ export FLASK_ENV=development
$ flask run
```

for Windows:
```
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

This will start the Flask application on `http://localhost:5000`.

### Using the application

To use the application, open your web browser and go to `http://localhost:5000`. This will take you to the home page, where you can upload an image file and get predictions from the pre-trained PyTorch model.

## Project structure

The project has the following structure:

```
.
├── app.py
├── static
├── templates
│ ├── home.html
│ └── results.html
├── labels.json
├── README.md
└── requirements.txt
```

Here's a brief overview of each file and directory:

- `app.py`: This is the main Flask application file. It contains the route definitions and the view functions that handle incoming requests.
- `static`: This directory contains the static assets used by the web application, such as CSS and JavaScript files. (Empty for now)
- `templates`: This directory contains the HTML templates used by the Flask views.
- `labels.json`: This file contains the mapping from class indices to label names.
- `README.md`: This file contains the documentation for the project.
- `requirements.txt`: This file contains the list of Python dependencies required to run the project.

## Acknowledgments

This project was inspired by the PyTorch Image Classification tutorial at https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html. The pre-trained model weights were downloaded from the PyTorch model zoo at https://pytorch.org/docs/stable/torchvision/models.html. 