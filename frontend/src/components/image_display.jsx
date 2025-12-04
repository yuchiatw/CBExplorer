import { useState, useEffect } from 'react';
export default function ImageDisplay({dataset, experiment}) {

    const [concepts, setConcepts] = useState([]);
    const [bits, setBits] = useState(Array(8).fill(0));
    const filename = bits.map(bit => bit.toString()).join('') + '.png';


    const toggleBit = (index) => {
        setBits(prev => {
            const next = [...prev];
            next[index] = prev[index] === 0 ? 1 : 0;
            return next;
        });
    };

    useEffect(() => {
        fetch("http://localhost:8000/concepts/"+dataset+"/"+experiment)
        .then(response => response.json())
        .then(data => {
            console.log("Fetched concepts:", data);
            
            // You can use the fetched data to set up your component state if needed
        })
        .catch(error => {
            console.error("Error fetching concepts:", error);
        });
    }, [dataset, experiment]);
    

    return (
        <div className="flex flex-row gap-4 items-center">
            <div className="flex flex-col gap-2">
                {bits.map((bit, idx) => (
                    <div key={idx} className="flex items-center justify-between gap-4">
                        <span className="text-gray-700">Attribute {idx+1}</span>

                        <label className="switch">
                            <input 
                                type="checkbox" 
                                checked={bit === 1}
                                onChange={() => toggleBit(idx)}
                            />
                            <span className="slider"></span>
                        </label>
                    </div>
                ))}
            </div>
            <div>
                <img src={path} alt-text="Displayed Image" style={{ maxWidth: '100%', height: 'auto' }} />
                <p className="text-sm text-gray-600">{filename}</p>
            </div>
            {/* <div>
                <svg>
                    {bits.map((bit, idx) => )}
                </svg>
                
            </div> */}

            {/* 8 toggle switches */}
            
            
        </div>            
    );
}