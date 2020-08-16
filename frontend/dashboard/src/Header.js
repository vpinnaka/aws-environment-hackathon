import React, { useState, useEffect } from "react";
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
    <AppBar className="header" position="static">
         <Toolbar>
      <Typography variant="h6" color="inherit" className="header__title">
        AWS Spheres
      </Typography>
      <Typography color="inherit" className="header__time">
        {dt}
      </Typography>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
