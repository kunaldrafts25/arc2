from scripts.audit_release import findings


def test_actual_credential_is_detected_without_returning_its_value():
    secret=b'local-sensitive-'+b'9'*32
    issues=findings('docs/example.md',b'prefix '+secret,1024,secret)
    assert issues==['matches project-local credential']
    assert secret.decode() not in str(issues)


def test_token_placeholder_is_allowed_but_realistic_token_is_blocked():
    assert findings('README.md',b'KGAT_'+b'x'*40,1024)==[]
    assert findings('docs/example.md',b'KGAT_'+b'A'*40,1024)==['Kaggle access token']


def test_weights_credentials_and_large_files_are_rejected():
    assert 'model weights are not part of the code release' in findings('adapter.safetensors',b'weights',1024)
    assert 'credential file' in findings('.kaggle/access_token',b'example',1024)
    assert 'file exceeds release size limit' in findings('output.txt',b'a'*20,10)


def test_private_keys_and_signed_urls_are_detected_without_echoing_them():
    data=b'-----BEGIN '+b'PRIVATE KEY-----'
    assert findings('bad.txt',data,1024)==['private key']
    data=b'https://example.invalid/?X-Goog-'+b'Signature=secret'
    assert findings('download.json',data,1024)==['signed download URL']
