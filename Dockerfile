FROM node:10 AS frontend-build

WORKDIR /usr/src/app
COPY dog_breed_identifier_web/ ./dog_breed_identifier_web/
RUN cd dog_breed_identifier_web && npm install && npm run build

FROM node:10 AS backend-build
WORKDIR /root/
COPY --from=frontend-build /usr/src/app/dog_breed_identifier_web/build ./dog_breed_identifier_web/build
COPY dog_breed_identifier_middleware/ ./dog_breed_identifier_middleware/
RUN cd dog_breed_identifier_middleware && npm install

EXPOSE 4001
CMD ["node", "./dog_breed_identifier_middleware/index.js"]