import React from 'react';
import Header from './Header.js';
import Graph from './Graph.js';
import StatusBar from './StausBar.js';

import './App.css';

function App() {
  return (
    <div className="app">
      <Header/>
      <StatusBar/>
       {/* StatusBar */}
      <Graph sensor_type='co2'/>
      <Graph sensor_type='co2'/>
      <Graph sensor_type='co2'/>
      <Graph sensor_type='co2'/>
    </div>
  );
}

export default App;
