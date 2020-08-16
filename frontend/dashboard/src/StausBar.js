import React from "react";
import {Alert, AlertTitle} from "@material-ui/lab";

function StausBar() {
  return (
    <div className="status-bar">
      <Alert severity="warning">
        <AlertTitle>Warning</AlertTitle>
        Sensor <strong>cos_1</strong> detected anamoly
      </Alert>
    </div>
  );
}

export default StausBar;
