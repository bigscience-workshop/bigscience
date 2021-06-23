# Workarounds

## Missing certificates

Sometimes, some certificates can be missing. It's possible to point to our own local versions of the certificates. You can simply copy them to `$six_ALL_CCFRWORK/etc/ssl/certs/` or any other relevant folder:
```bash
export CURL_CA_BUNDLE=$six_ALL_CCFRWORK/etc/ssl/certs/ca-certificates.crt
```
