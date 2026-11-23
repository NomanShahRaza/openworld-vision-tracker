$HostName = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$Port = if ($env:PORT) { $env:PORT } else { "8000" }
python -m uvicorn app.main:app --host $HostName --port $Port
