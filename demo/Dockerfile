# syntax=docker/dockerfile:1

FROM nginx

WORKDIR /demo

COPY nginx.conf /etc/nginx/conf.d/
COPY . /var/www/html/

EXPOSE 8080