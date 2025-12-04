import { useState, useEffect } from 'react';
import BarChart from './barchart';
export default function ManipulateImage({ dataset, experiment, concepts, bits, setBits, seed }) {

    const [imageSrc, setImageSrc] = useState('');
    const [conceptLogits, setConceptLogits] = useState([]);

    const toggleBit = (index) => {
        setBits(prev => {
            const next = [...prev];
            next[index] = prev[index] === 0 ? 1 : 0;
            return next;
        });
    };

    useEffect(() => {
        fetch("http://localhost:8000/manipulate/" + experiment + "/" + dataset + "/" + seed + '?bit=' + bits.join(''))
            .then(response => response.json())
            .then(data => {
                console.log("Image generated with seed:", seed);
                if (data.image) {
                    setImageSrc(data.image);
                }
                if (data.concept_probs) {
                    setConceptLogits(data.concept_probs);
                }
            })
            .catch(error => {
                console.error("Error generating image:", error);
            });
    }, [dataset, experiment, seed, bits, bits]);


    return (
        <div className="flex flex-row gap-4 items-center">
            <div>
                <img src={imageSrc} alt="Generated" style={{ maxWidth: '256px', height: 'auto' }} />
            </div>
            <div className="flex flex-col gap-2">
                {bits.map((bit, idx) => (
                    <div key={idx} className="flex items-center justify-between gap-4">
                        <span className="text-gray-700">{concepts[idx]}</span>

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
                <BarChart concepts={concepts} conceptLogits={conceptLogits} />
            </div>


        </div>
    );
}