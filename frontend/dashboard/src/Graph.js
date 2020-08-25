import 'date-fns';
import React, { useState, useEffect } from "react";
import { formatISO } from 'date-fns';
import Plot from "react-plotly.js";
import { API, graphqlOperation } from "aws-amplify";
import DateFnsUtils from '@date-io/date-fns';
import { MuiPickersUtilsProvider, KeyboardDatePicker } from "@material-ui/pickers";
import StatusBar from "./StausBar";
import "./Graph.css";

export default function Graph({ sensorFullName, gqlEnpoint, galQueryName, setanamolyState }) {
  const [sensorValues, setsensorValues] = useState([]);
  //const [anamolyState, setanamolyState] = useState({ value: false });
  const [selectedDate, setSelectedDate] = React.useState(new Date());
  const todayDateStr = formatISO(new Date(), { representation: 'date' })

  useEffect(() => {
    var selectedDateStr = formatISO(selectedDate, { representation: 'date' });
    if (selectedDateStr == todayDateStr){
      const interval = setInterval(() => {
        fetchSensordata();
      }, 1000);
      return () => clearInterval(interval);
    }else{
      fetchSensordata();
    }
    
  }, [sensorFullName, selectedDate]);

  async function fetchSensordata() {
    try {
      const apiData = await API.graphql(
        graphqlOperation(gqlEnpoint, {
          filter:{
            timestamp:{
              contains: formatISO(selectedDate, { representation: 'date' })
            }
          },
          limit: 10000,
        })
      );

      setsensorValues(() => {
        let sensorData = {
          anomalies: [],
        };
        for (let key in apiData.data[galQueryName].items) {
          let item = apiData.data[galQueryName].items[key];
          if (!(item.name in sensorData)) {
            sensorData[item.name] = [];
          }
          if (item.anamoly === true) {
            var selectedDateStr = formatISO(selectedDate, { representation: 'date' });
            if (selectedDateStr == todayDateStr){
              setanamolyState({ value: true, sensorName: sensorFullName});
            }
            
            sensorData["anomalies"].push(item.timestamp);
          }
          sensorData[item.name].push([item.timestamp, item.value]);
        }
        return sensorData;
      });
    } catch (error) {
      console.log(error);
    }
  }

  function getAnamolusData(anomalies) {
    let shapes = [];
    anomalies.sort().forEach((date) => {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: date,
        y0: 0,
        x1: date,
        y1: 1,
        line: {
          color: "red",
          width: 3,
          opacity: 0.2,
        },
        name: "anamoly",
      });
    });
    return shapes;
  }

  function getPlotdata(sensorValues) {
    let values = [];
    for (const [sensorName, sensorValue] of Object.entries(sensorValues)) {
      if (sensorName === "anomalies") {
        continue;
      }
      let xs = [];
      let ys = [];
      sensorValue.sort().forEach((element) => {
        xs.push(element[0]);
        ys.push(element[1]);
      });

      values.push({
        x: xs,
        y: ys,
        type: "scatter",
        mode: "lines",
        name: sensorName,
      });
    }
    return values;
  }

  const handleDateChange = (date) => {
    setSelectedDate(date);
    setanamolyState({ value: false, sensorName: ""});
  };

  return (
    <div className="graph">
      <div>
        <h3>{sensorFullName} sensor data</h3>
        <MuiPickersUtilsProvider utils={DateFnsUtils}>
        <KeyboardDatePicker
          disableToolbar
          variant="inline"
          format="MM/dd/yyyy"
          margin="normal"
          id="date-picker-inline"
          label="Values record date"
          value={selectedDate}
          onChange={handleDateChange}
          KeyboardButtonProps={{
            "aria-label": "change date",
          }}
        />
        </MuiPickersUtilsProvider>
      </div>
      <Plot
        data={getPlotdata(sensorValues)}
        layout={{
          shapes: sensorValues.anomalies
            ? getAnamolusData(sensorValues.anomalies)
            : [],
          width: 1500,
          height: 700,
        }}
        config={{ responsive: true }}
      />
      <p>
        <strong>Note:</strong> Veritcal lines indicate anaomoly
      </p>
    </div>
  );
}
