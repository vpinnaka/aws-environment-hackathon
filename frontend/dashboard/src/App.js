import React from "react";
import Header from "./Header";
import SideBar from "./SideBar";
import DashBoard from "./DashBoard";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import Amplify from 'aws-amplify';
import awsconfig from './aws-exports';
import "./App.css";

Amplify.configure(awsconfig);

function App() {
  return (
    <div className="app">
      <Router>
        <Header />
        <div className="app__body">
          <SideBar />
          <div className="app__content">
            <Switch>
              <Route path="/sensor/:sensorId">
                <DashBoard />
              </Route>
              <Route path="/">
                <h1>Welcome</h1>
              </Route>
            </Switch>
          </div>
        </div>
      </Router>
    </div>
  );
}

export default App;
