function [g, data] = cpp2matG(gCPP, dataCPP)

pdDims = [];
for i=1:length(gCPP.bdry_type)
  switch(gCPP.bdry_type(i))
    case 0
    case 1
      % @addGhostPeriodic;
      pdDims = [pdDims; i];
    case 2
      % @addGhostExtrapolate;
    otherwise
      error('Unknown Boundary Type!')
  end
end

process = false;
low_mem = true;
g = createGrid(gCPP.min, gCPP.max, gCPP.N, pdDims, process, low_mem);

if nargin == 2 && nargout == 2
  % Data
  data = zeros(size(dataCPP{1}));
  clns = repmat({':'}, 1, g.dim);
  for i = 1:length(dataCPP)
    data(clns{:}, i) = dataCPP{i};
    dataCPP{i} = [];
  end
end
end