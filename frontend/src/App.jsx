import { useState, useEffect } from 'react'
import ImageDisplay from './components/image_display'
import AltImageDisplay from './components/manipulate_image';
import './App.css'
import ListButton from './components/list_button';
import ConceptMapper from './utils/concept_mapper';

const colorcode = ['steelblue', 'pink', 'grey'];
function App() {
  const datalist = ['cub', 'celebahq']
  const expList = ['cbae_stygan2', 'cc_stygan2']
  const [dataset, setDataset] = useState('celebahq')
  const [experiment, setExperiment] = useState('cbae_stygan2')
  const [concepts, setConcepts] = useState([]);
  const [draftValue, setDraftValue] = useState(42);
  const [value, setValue] = useState(42); // This triggers useEffect
  const [seed, setSeed] = useState(42);

  const [imageSrc, setImageSrc] = useState('');
  const [th, setTh] = useState(0.1);

  const [originalImageSrc, setOriginalImageSrc] = useState('');
  const [conceptLogits, setConceptLogits] = useState([]);
  const [altLogits, setAltLogits] = useState([]);
  const [bits, setBits] = useState([]);

  const [expanded, setExpanded] = useState(false);
  const imagePath1 = "/small-group-4.png";
  const imagePath2 = "/extend-group-2.png";
  const mappedConcepts = ConceptMapper(dataset, concepts);
  function getButtonClass(logit) {
    if (logit - 0.5 < th && logit - 0.5 > -th) return "button-unavailable";
    if (logit >= 0.5) return "button-positive";
    return "button-negative";
  }

  const toggleBit = (index) => {
    setBits(prev => {
      const next = [...prev];
      next[index] = prev[index] === 0 ? 1 : 0;
      return next;
    });
  };

  useEffect(() => {
    fetch("http://localhost:8000/concepts/" + experiment + "/" + dataset)
      .then(response => response.json())
      .then(data => {
        console.log("Fetched concepts:", data);
        if (data.concepts) {
          setConcepts(data.concepts);
        }
      })
      .catch(error => {
        console.error("Error fetching concepts:", error);
      });
    fetch("http://localhost:8000/generate/" + experiment + "/" + dataset + "/" + seed)
      .then(response => response.json())
      .then(data => {
        console.log("Image generated with seed:", seed);
        if (data.image) {
          setOriginalImageSrc(data.image);
        }
        if (data.concept_probs) {
          setConceptLogits(data.concept_probs);
        }
        if (data.concept_values) {
          setBits(data.concept_values);
          console.log("Initial concept values:", data.concept_values);
        }

      })
      .catch(error => {
        console.error("Error generating image:", error);
      });
  }, [dataset, experiment, seed]);

  useEffect(() => {
    
    fetch("http://localhost:8000/manipulate/" + experiment + "/" + dataset + "/" + seed + '?bit=' + bits.join(''))
      .then(response => response.json())
      .then(data => {
        console.log("Image generated with seed:", seed);
        if (data.image) {
          setImageSrc(data.image);
        }
        if (data.concept_probs) {
          setAltLogits(data.concept_probs);
        }

      })
      .catch(error => {
        console.error("Error generating image:", error);
      });
  }, [dataset, experiment, seed, bits]);



  return (
    <div className="flex flex-col min-h-screen">
      <header className="flex flex-row header-container bg-zinc-900 text-center text-white py-4 px-8 text-xl">
        <h2 className="text-left text-2xl">CBExplorer</h2>
        <label htmlFor="data-select" className="mx-2">
          <select
            id='data-select'
            className="bg-gray-200 text-black p-2 rounded mx-2"
            value={dataset}
            onChange={e => setDataset(e.target.value)}
          >
            {datalist.map((data, index) => (
              <option
                key={index}
                value={data}
              >
                {data}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor="experiment-select" className="mx-2">
          <select
            id='experiment-select'
            className="bg-gray-200 text-black p-2 rounded mx-2"
            value={experiment}
            onChange={e => setExperiment(e.target.value)}
          >
            {expList.map((exp, index) => (
              <option
                key={index}
                value={exp}
              >
                {exp}
              </option>
            ))}
          </select>
        </label>
      </header>
      <main className="flex flex-row items-start justify-center">
        <div className="flex flex-row mr-5 mt-70 gap-4">

          <input
            type="text" className="underline-input"
            placeholder="Type here..." value={draftValue}
            onChange={e => setDraftValue(e.target.value)}
          />
          <button className="generate-button" onClick={() => setSeed(draftValue)}>Generate</button>

        </div>
        <div className="flex flex-row items-start gap-0 mt-20">
          <div
            className='mt-30'

          >
            {expanded ? (
              <div className="flex flex-col gap-0">
                <img src={imagePath2} alt="Group" onClick={() => setExpanded(!expanded)} />

              </div>
            ) : (
              <div className="flex flex-col gap-0">
                <img src={imagePath1} alt="Group" onClick={() => setExpanded(!expanded)} />
              </div>
            )}
            <ListButton
              expanded={expanded}
              altLogits={altLogits}
              concepts={concepts}
              bits={bits}
              setBits={setBits}
              dataset={dataset}
            />
          </div>
          <div className='flex flex-col gap-5'>
            <ImageDisplay
              mappedConcepts={mappedConcepts}
              concepts={concepts}
              imageSrc={originalImageSrc}
              conceptLogits={conceptLogits}
            />
            <ImageDisplay
              mappedConcepts={mappedConcepts}
              concepts={concepts}
              imageSrc={imageSrc}
              conceptLogits={altLogits}
            />
          </div>
        </div>

      </main>
      <footer></footer>
    </div>

  )
}

export default App;
