# SPI Web Demo
Spare Parts Identification Web Demo

## Run
```sh
sudo docker build . -t spi_demo
sudo docker run -d --name spi_demo --network=host -it spi_demo # --host 'localhost' --port 8081
```

