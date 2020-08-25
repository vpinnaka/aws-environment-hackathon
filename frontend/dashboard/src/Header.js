import React, { useState, useEffect } from "react";
import AccessTimeIcon from "@material-ui/icons/AccessTime";
import { AppBar, Typography, Toolbar } from "@material-ui/core";
import "./Header.css";

function Header() {
  const [dt, setDt] = useState(new Date().toLocaleString());

  useEffect(() => {
    let secTimer = setInterval(() => {
      setDt(
        new Date().toLocaleString([], {
          year: "numeric",
          month: "numeric",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    }, 1000);

    return () => clearInterval(secTimer);
  }, []);

  return (
    <div className="header" position="static">
      <div className="header__left">
        <h3>
          AWS Spheres
        </h3>
      </div>
      <div className="header__center">
        <AccessTimeIcon />
        <h3>
          {dt}
        </h3>
      </div>
      <div className="header__right"></div>
    </div>
  );
}

export default Header;
