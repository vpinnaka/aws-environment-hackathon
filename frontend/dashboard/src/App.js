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
      <Graph />
    </div>
  );
}

export default App;
