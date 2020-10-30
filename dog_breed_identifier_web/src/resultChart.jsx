import React, {Component} from 'react'
import {Pie} from '@reactchartjs/react-chart.js'

const data = {
    labels: [],
    datasets: [
        {
            data: [],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)',
            ],
            borderWidth: 1,
        },
    ],
}

export default class ResultChart extends Component {
    convertResultToArray = (result) => {
        data.labels = []
        data.datasets[0].data = []

        result = result.split("\n")
        result.forEach(function (item, index) {
            item = item.split(":")

            var i, frags = item[0].split('_');
            for (i = 0; i < frags.length; i++) {
                frags[i] = frags[i].charAt(0).toUpperCase() + frags[i].slice(1);
            }

            let name = frags.join(' ');
            let value = parseFloat(item[1]) * 100

            data.labels.push(name)
            data.datasets[0].data.push(value.toFixed(2))
        })
        return data
    }

    render() {
        return (
            <Pie data={this.convertResultToArray(this.props.result)}/>
        )
    }
}