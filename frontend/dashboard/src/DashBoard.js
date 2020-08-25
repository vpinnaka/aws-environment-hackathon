import React, { useState } from "react";
import Graph from "./Graph";
import "./DashBoard.css";
import {
  listCo2s,
  listTemparatures,
  listDewpoints,
  listRelativehumiditys,
} from "./graphql/queries";
import { useParams } from "react-router-dom";
import StatusBar from "./StausBar";

function DashBoard() {
  const { sensorId } = useParams();
  const [anamolyState, setanamolyState] = useState({
    value: false,
    sensorName: "",
  });

  const gqlEnpointMap = {
    co2: {
      sensorFullName: "CO2",
      gqlEnpoint: listCo2s,
      galQueryName: "listCo2s",
    },
    temp: {
      sensorFullName: "Temperature",
      gqlEnpoint: listTemparatures,
      galQueryName: "listTemparatures",
    },
    rel_hum: {
      sensorFullName: "Relative humidity",
      gqlEnpoint: listRelativehumiditys,
      galQueryName: "listRelativehumiditys",
    },
    dew: {
      sensorFullName: "Dewpoint",
      gqlEnpoint: listDewpoints,
      galQueryName: "listDewpoints",
    },
  };
  return (
    <div className="dashboard">
      <div>
        {anamolyState.value == false ||  anamolyState.addressed == false? (
          <span></span>
        ) : (          
            <StatusBar
              title="Warning"
              severity="error"
              sersorName={anamolyState.sensorName}
              setanamolyState={setanamolyState}
            />
        )}
      </div>
      <Graph
        sensorFullName={gqlEnpointMap[sensorId].sensorFullName}
        gqlEnpoint={gqlEnpointMap[sensorId].gqlEnpoint}
        galQueryName={gqlEnpointMap[sensorId].galQueryName}
        setanamolyState={setanamolyState}
      />
    </div>
  );
}

export default DashBoard;
