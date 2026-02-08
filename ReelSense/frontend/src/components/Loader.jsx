import React from 'react';
import MovieCard from './MovieCard';

function Loader() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
      {Array.from({ length: 10 }).map((_, index) => (
        <MovieCard key={index} isSkeleton />
      ))}
    </div>
  );
}

export default Loader;