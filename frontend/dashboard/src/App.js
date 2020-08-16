import React from 'react';
import Header from './Header.js';
import Graph from './Graph.js'

import './App.css';

function App() {
  return (
    <div className="app">
      <Header/>
       
       {/* StatusBar */}
      <Graph />
    </div>
  );
}

export default App;
