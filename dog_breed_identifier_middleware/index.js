const express = require("express")
const http = require("http")
const socketIo = require("socket.io")
const redis = require("redis")
const cors = require("cors");
const path = require('path');

const socket_port = 4001

const app = express()

var corsOptions = {
    origin: "*"
};

app.use(cors(corsOptions));
app.use(express.static(path.join(__dirname, '../dog_breed_identifier_web/build')));
app.get("/", (req, res) => {
	res.sendFile(path.join(__dirname, '../dog_breed_identifier_web/build/index.html'))
})


const server = http.createServer(app);

const io = socketIo(server)
const redisURL = "redis://0.0.0.0:6379"

function onConnect(socket) {
	const subscriber = redis.createClient(redisURL);
	subscriber.on("message", (channel, message) => {
		console.log(message)
	})

	socket.on("subscribe", function() {
		channel = 'identifier_result'
		subscriber.subscribe(channel)
		console.log("Subscribed to channel: " + channel)
	})
}

io.on("connection", onConnect)
server.listen(socket_port, () => console.log(`Listening on port ${socket_port}`))