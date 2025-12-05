import { useState, useEffect } from 'react';
import BarChart from './barchart';

const colorcode = ['steelblue', 'pink', 'grey'];
export default function ManipulateImage({ concepts, imageSrc, conceptLogits }) {

    // const [imageSrc, setImageSrc] = useState('');
    // const [conceptLogits, setConceptLogits] = useState([]);
    // const [th, setTh] = useState(0.1);

    // const toggleBit = (index) => {
    //     setBits(prev => {
    //         const next = [...prev];
    //         next[index] = prev[index] === 0 ? 1 : 0;
    //         return next;
    //     });
    // };
    // function getButtonClass(logit) {
    //     if (logit - 0.5 < th && logit - 0.5 > -th) return "button-unavailable";
    //     if (logit >= 0.5) return "button-positive";
    //     return "button-negative";
    // }


    // useEffect(() => {
    //     fetch("http://localhost:8000/manipulate/" + experiment + "/" + dataset + "/" + seed + '?bit=' + bits.join(''))
    //         .then(response => response.json())
    //         .then(data => {
    //             console.log("Image generated with seed:", seed);
    //             if (data.image) {
    //                 setImageSrc(data.image);
    //             }
    //             if (data.concept_probs) {
    //                 setConceptLogits(data.concept_probs);
    //             }
    //         })
    //         .catch(error => {
    //             console.error("Error generating image:", error);
    //         });
    // }, [dataset, experiment, seed, bits]);


    return (
        <div className="flex flex-row gap-4 items-center">
            <div>
                <img src={imageSrc} alt="Generated" style={{ maxWidth: '256px', height: 'auto' }} />
            </div>
            

            <div>
                <BarChart concepts={concepts} conceptLogits={conceptLogits} />
            </div>


        </div>
    );
}