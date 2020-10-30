import React, {Component} from 'react'
import {DropzoneArea} from 'material-ui-dropzone'
import {Button} from "@material-ui/core";
import axios from 'axios';

class Uploader extends Component {
    constructor(props) {
        super(props);
        this.state = {
            files: []
        };
    }

    handleChange = (files) => {
        this.setState({files: files})
    }

    handleSubmit = () => {
        const files = Array.from(this.state.files)
        const data = new FormData()

        files.forEach((file, i) => {
            data.append(i, file)
        })

        console.log(data.values())

        axios.post("http://localhost:5000/upload", data, {
            headers: {"Content-Type": "multipart/form-data"},
        }).then(res => {
            console.log(res)
        }).catch(err => {
            console.log(err)
        })
    }

    render() {
        return (
            <div>
                <DropzoneArea
                    onChange={this.handleChange}
                    maxFileSize={10000000}
                    dropzoneText={"Upload your dog picture"}
                />
                <div style={{margin: 20, display: "flex", alignItems: "center", justifyContent: "center"}}>
                    <Button variant="contained" color="primary" onClick={this.handleSubmit}>Identify</Button>
                </div>
            </div>
        )
    }
}

export default Uploader;