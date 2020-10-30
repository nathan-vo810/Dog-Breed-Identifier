import React, {Component} from 'react'
import {DropzoneArea} from 'material-ui-dropzone'
import {Button, CircularProgress, Typography} from "@material-ui/core";
import axios from 'axios';
import socketIOClient from "socket.io-client";
import ResultChart from "./resultChart"

class Uploader extends Component {
    constructor(props) {
        super(props);
        this.state = {
            files: [],
            pendingResult: false,
            result: null,
            noDog: false
        };
        this.socket = socketIOClient("http://localhost:4001")
        this.socket.emit("subscribe", 0)
    }

    handleChange = (files) => {
        this.setState({files: files})
    }

    handleSubmit = () => {
        this.setState({result: null})
        const files = Array.from(this.state.files)
        const data = new FormData()

        files.forEach((file, i) => {
            data.append(i, file)
        })

        axios.post("http://localhost:5000/upload", data, {
            headers: {"Content-Type": "multipart/form-data"},
        }).then(res => {
            this.setState({pendingResult: true})
        }).catch(err => {
            console.log(err)
        })
    }


    displayResult = (data) => {
        if (data === "NO DOG!") {
            this.setState({noDog: true, pendingResult: false})
        } else {
            this.setState({pendingResult: false, noDog: false, result: data})
        }
    }

    componentDidMount() {
        this.socket.on("result", this.displayResult)
    }

    render() {
        return (
            <div style={{width: "40vw"}}>
                <DropzoneArea
                    onChange={this.handleChange}
                    maxFileSize={10000000}
                    dropzoneText={"Upload your dog picture"}
                    filesLimit={1}
                />
                <div style={{
                    margin: 20,
                    display: "flex",
                    alignItems: "center",
                    flexDirection: "column",
                    justifyContent: "center"
                }}>
                    {this.state.pendingResult ?
                        <CircularProgress/> :
                        <Button variant="contained" color="primary" onClick={this.handleSubmit}>Identify</Button>
                    }
                    <br/>
                </div>

                {this.state.result ?
                    <div style={{backgroundColor: "#FFFFFF", padding: 10}}>
                        <ResultChart result={this.state.result}/>
                    </div> :
                    this.state.noDog ?
                        <Typography variant="h5">No dog detected! Please try another picture!</Typography> :
                        <div/>
                }
            </div>
        )
    }
}

export default Uploader;