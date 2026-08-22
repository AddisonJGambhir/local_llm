FROM node:22-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY tsconfig.json ./
COPY src ./src
RUN npx tsc

ENV CHATROOM_HOST=0.0.0.0 \
    CHATROOM_PORT=8790 \
    CHATROOM_DATA=/data
VOLUME /data
EXPOSE 8790

CMD ["node", "dist/server.js"]
