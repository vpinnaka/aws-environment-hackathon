import React, { useState, useEffect } from "react";
import Email from "@material-ui/icons/Email";
import "./SideBar.css";
import SideBarOption from "./SideBarOption";

function SideBar() {
  const [sensors, setSensors] = useState([]);

  useEffect(() => {
    setSensors([
      { id: "co2", name: "CO2" },
      { id: "temp", name: "Temparature" },
      { id: "dew", name: "Dewpoint" },
      { id: "rel_hum", name: "Relative humidity" },
    ]);
  }, []);

  return (
    <div className="sidebar">
      <div className="sidebar__header">
        <h2>Signed in User</h2>

        <h3>
          <Email />
          user@amazon.com
        </h3>
      </div>
      <div className="sidebar__body">
        {sensors.map((sensor) => {
          return <SideBarOption title={sensor.name} id={sensor.id} />;
        })}
        <hr />
      </div>
    </div>
  );
}

export default SideBar;
