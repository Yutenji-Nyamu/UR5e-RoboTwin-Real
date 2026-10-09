import json,math,os,tempfile,unittest
from pathlib import Path
from supervision import serializable_log,process_identity,identity_matches,refresh_progress,completed_run
class SupervisionRegression(unittest.TestCase):
    def test_compact_mode_marker_preserves_numeric_metrics(self):
        raw=json.loads('{"step":3000,"anneal/dense_bias":-Infinity,"loss/total":0.93,"grad_norm":2.5}')
        out=serializable_log(raw);json.dumps(out,allow_nan=False)
        self.assertEqual(out['anneal/dense_bias'],'-inf');self.assertEqual(out['loss/total'],raw['loss/total']);self.assertEqual(raw['anneal/dense_bias'],-math.inf)
        self.assertNotIn('unexpected_nonfinite_metrics',out)
    def test_nonfinite_loss_stays_visible(self):
        out=serializable_log({'loss/total':math.nan});json.dumps(out,allow_nan=False)
        self.assertEqual(out['unexpected_nonfinite_metrics'],['loss/total']);self.assertEqual(out['loss/total'],'nan')
    def test_partial_log_and_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'train_log.jsonl').write_text('{"step":3000,"anneal/dense_bias":-Infinity,"loss/total":0.9}\n{"step":')
            state=refresh_progress({},p,p);json.dumps(state,allow_nan=False);self.assertEqual(state['last_log']['step'],3000)
            c=p/'checkpoints/step_0003000';c.mkdir(parents=True);self.assertFalse(completed_run(p,3000))
            (c/'complete.json').write_text('{"step":3000}');self.assertTrue(completed_run(p,3000));self.assertFalse(completed_run(p,30000))
    def test_process_identity_rejects_wrong_start_or_uid(self):
        x=process_identity(os.getpid());self.assertTrue(identity_matches(x,x))
        self.assertFalse(identity_matches(x,dict(x,starttime='0')));self.assertFalse(identity_matches(x,dict(x,uid=-1)))
if __name__=='__main__':unittest.main()
