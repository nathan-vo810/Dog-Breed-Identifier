import React from 'react';
import Uploader from "./Uploader";
import {Paper, Typography} from "@material-ui/core"
import withStyles from "@material-ui/core/styles/withStyles";
import Image from './bg.jpg'

const styles = () => ({
    root: {
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        backgroundImage: `url(${Image})`,
    }
})

class App extends React.Component {
    render() {

        const {classes} = this.props;

        return (
            <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                <Typography variant="h4" style={{marginTop: 20}}> Dog Breed Identifier </Typography><br/>
                <Paper className={classes.root} style={{margin: 20, padding: 20}}>
                    <br/><br/>
                    <div style={{margin: 10, marginTop: 30, width: "100%"}}>
                        <Uploader/>
                    </div>
                </Paper>
            </div>
        );
    }
}

export default withStyles(styles)(App);