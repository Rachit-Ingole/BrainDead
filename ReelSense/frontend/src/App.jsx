import React, { useState } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import MovieGrid from './components/MovieGrid';
import Loader from './components/Loader';

function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRecommendations = async () => {

    if (!userId.trim() || isNaN(userId)) {
      setError('Please enter a valid numeric User ID.');
      return;
    }

    setLoading(true);
    setError(null);
    setRecommendations([]);

    try {
     
      const response = await axios.get('/recommendations', {
        params: {
          user_id: parseInt(userId),
          top_k: 10
        }
      });
      setRecommendations(response.data);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch recommendations. Ensure the backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar
        userId={userId}
        setUserId={setUserId}
        onGetRecommendations={fetchRecommendations}
      />
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <section className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-red-600 mb-4">
            Movies You’ll Love
          </h1>
          <p className="text-lg md:text-xl text-gray-300">
            Powered by AI + Explainability
          </p>
        </section>

        {/* Recommendation Grid or Loading/Error */}
        {loading ? (
          <Loader />
        ) : error ? (
          <div className="text-center text-red-500 text-xl">{error}</div>
        ) : (
          <MovieGrid recommendations={recommendations} />
        )}
      </main>
    </div>
  );
}

export default App;