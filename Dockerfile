FROM python:3.12

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# create base folder that we will use for our program
WORKDIR /app

# python file to our docker container
COPY . /app

# install python venv
RUN apt update && apt install -y python3-venv python3-opencv

# create virtual environment
RUN python3 -m venv venv

# install dependencies
RUN ./venv/bin/pip install --upgrade pip
RUN ./venv/bin/pip install -r requirements.txt

# replace the deprecated line
RUN sed -i 's/w, h = self.font.getsize(label)/_, _, w, h = self.font.getbbox(label)/g' venv/lib/python3.12/site-packages/yolov5/utils/plots.py

# download model weights
RUN wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1so1UyzoTugUc0-7oFY6e65H5QsOq6sfw" -O spi_demo_yolov5l.pt

# command to run
ENTRYPOINT ["./venv/bin/python", "-u", "src/app.py"]
