import React from "react";
import { useHistory } from "react-router-dom";
import "./SideBarOption.css";

function SideBarOption({ title, id }) {
    const history = useHistory();

    const selectSensor = () =>{
        if (id){
            history.push(`/sensor/${id}`);
        }
    };

    return (
        <div className="sidebaroption" onClick={selectSensor}>
            <h3>{title}</h3>
        </div>
    )
}

export default SideBarOption
