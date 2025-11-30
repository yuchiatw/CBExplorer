import { useState } from 'react'
import RenderOptions from './components/options'
import ImageDisplay from './components/image_display'
import Generator from './components/generator'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="flex flex-col min-h-screen">
      <header className="flex flex-row header-container bg-zinc-900 text-center text-white py-4 px-8 text-xl">
        <h2 className="text-left text-2xl">CBExplorer</h2>
        <label htmlFor="bar-select" className="mx-2">
          <select id = 'bar-select' className="bg-gray-200 text-black p-2 rounded mx-2">
              <RenderOptions />
          </select>
        </label>
      </header>
      <main>
        <div className="flex flex-col items-center justify-center p-10">
          <ImageDisplay />
        </div>
      </main>
      <footer></footer>
    </div>

  )
}

export default App
