import React from 'react'
import Plot from 'react-plotly.js';
import sensordata from './data/temp.json';
import './Graph.css'

export default function Graph({sensor_type}) {
    const values = sensordata[0]

    var layout = {
        // to highlight the timestamp we use shapes and create a rectangular
        shapes: [
            // 1st highlight during Feb 4 - Feb 6
            {
                type: 'rect',
                // x-reference is assigned to the x-values
                xref: 'x',
                // y-reference is assigned to the plot paper [0,1]
                yref: 'paper',
                x0: values.timestamp[10],
                y0: 0,
                x1: values.timestamp[15],
                y1: 1,
                fillcolor: 'red',
                opacity: 0.5,
                line: {
                    width: 0
                }
            },
            // 2nd highlight during Feb 20 - Feb 23
            {
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: values.timestamp[40],
                y0: 0,
                x1: values.timestamp[50],
                y1: 1,
                fillcolor: 'red',
                opacity: 0.5,
                line: {
                    width: 0
                }
            }
            
        ],
        width: 1500,
        height: 700,
    }
    return (
        <div className='graph'>
            <p>{sensor_type} sensor data</p>
            <Plot
            
        data={[
          {
            x: values.timestamp,
            y: values[sensor_type + '_1'],
            type: 'scatter',
            mode: 'lines',
            name:sensor_type + '_1',
            
          },
          {
            x: values.timestamp,
            y: values[sensor_type + '_2'],
            type: 'scatter',
            mode: 'lines',
            name:sensor_type + '_2',
            
          },
          {
            x: values.timestamp,
            y: values[sensor_type + '_3'],
            type: 'scatter',
            mode: 'lines',
            name:sensor_type + '_3',
            
            
          },
          {
            x: values.timestamp,
            y: values[sensor_type + '_4'],
            type: 'scatter',
            mode: 'lines',
            name:sensor_type + '_4',
            
          },
          
        ]}
        layout={layout}
        config = {{responsive: true}}
      />
        </div>
    )
}
