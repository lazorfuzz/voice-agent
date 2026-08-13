set -e

container run \
  --name livekit \
  -v "$(pwd)/livekit.yaml:/livekit.yaml" \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 50000-50100:50000-50100/udp \
  --rm \
  livekit/livekit-server:latest \
  --config /livekit.yaml
