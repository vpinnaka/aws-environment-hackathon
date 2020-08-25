import React from "react";
import { Alert, AlertTitle } from "@material-ui/lab";
import Button from "@material-ui/core/Button";

function StausBar({ title, severity, sersorName, setanamolyState }) {
  function updateAnamoly(){
    setanamolyState({ addressed: true});
  }

  return (
    <div className="status-bar">
      <Alert severity={severity}>
        <AlertTitle>{title}</AlertTitle>
        <div className="status-bar__body">
          <h3>
            Anamoly detected in <strong>{sersorName}</strong> sensors
          </h3>
          <Button variant="contained" color="secondary" onClick={updateAnamoly}>
            Addressed
          </Button>
        </div>
      </Alert>
    </div>
  );
}

export default StausBar;
