#Copyright July 20201 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets.info import DatasetInfo
from datasets.features import PandasArrayExtensionArray, PandasArrayExtensionDtype, Features, Value, cast_to_python_objects, pandas_types_mapper
from datasets import utils, Dataset
from datasets.splits import NamedSplit
from datasets.arrow_writer import ArrowWriter, OptimizedTypedSequence
import os
import json
from pathlib import Path
from datasets.utils.typing import PathLike
from datasets.arrow_dataset import transmit_format# , replayable_table_alteration
from transformers import PreTrainedModel, PretrainedConfig
import copy
import shutil
from datasets.fingerprint import (
    fingerprint_transform,
    generate_fingerprint,
    generate_random_fingerprint,
    get_temporary_cache_files_directory,
    is_caching_enabled,
    update_fingerprint,
)
from datasets.dataset_dict import DatasetDict
from torch import nn
import pickle

import glob, shutil, os, time
import indexed_gzip as igzip
#import zstandard, io
#from gzip_stream import GZIPCompressedStream
import  fsspec.compression

from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import dataset
import six
from six.moves.urllib.parse import parse_qs, urlparse


### NOTE: dataset is a different package than datasets. We are using both packages.


### We want to have mutliple types of storage that ideally can be
### transported as a file transfer with an arrow dataset. So if we
### have <signature>.arrow, we may have fts_<signature>.db (for full
### text indexing) and db_<signature>.db (sqlite database), and
### <siganture>.mmap (mmap file reprsenting a tensor), and
### <singature>.igz (if we wish to store some portion of the text
### columns in igzip format for compression and legacy purposes.

def is_contiguous(arr):
        start = None
        prev = None
        contiguous=True
        for i in arr:
          if start is None:
            start = i
          if prev is None or i == prev+1:
            prev = i
            continue
          contiguous = False
          break
        return contiguous, start, i+1

class TableExt(dataset.Table):


    def find(self, *_clauses, **kwargs):
        """Perform a simple search on the table similar to
        dataset.Table's find, except: optionally gets a result only
        for specific columns by passing in _columns keyword.

        # TODO, full text search

        Simply pass keyword arguments as ``filter``.
        ::

            results = table.find(country='France')
            results = table.find(country='France', year=1980)

        Using ``_limit``::

            # just return the first 10 rows
            results = table.find(country='France', _limit=10)

        You can sort the results by single or multiple columns. Append a minus
        sign to the column name for descending order::

            # sort results by a column 'year'
            results = table.find(country='France', order_by='year')
            # return all rows sorted by multiple columns (descending by year)
            results = table.find(order_by=['country', '-year'])

        To perform complex queries with advanced filters or to perform
        aggregation, use :py:meth:`db.query() <dataset.Database.query>`
        instead.
        """


        if not self.exists:
            return iter([])

        _fts = kwargs.pop('_fts', None)
        _columns = kwargs.pop('_columns', None)
        _limit = kwargs.pop('_limit', None)
        _offset = kwargs.pop('_offset', 0)
        order_by = kwargs.pop('order_by', None)
        _streamed = kwargs.pop('_streamed', False)
        _step = kwargs.pop('_step', QUERY_STEP)
        if _step is False or _step == 0:
            _step = None

        order_by = self._args_to_order_by(order_by)
        args = self._args_to_clause(kwargs, clauses=_clauses)

        if _fts:
            # we could run against a local sqlite database and join manually using a list of id's
            res = self.fts_db.executable.execute(f"""SELECT id, rank
                              FROM {table_name}_idx
                              WHERE {column} MATCH {fts_q}
                              ORDER BY rank
                              LIMIT {_limit}""").fetchall()

        if columns is None:
            query = self.table.select(whereclause=args,
                                  limit=_limit,
                                  offset=_offset)
        else:
            query = self.table.select(columns, whereclause=args,
                                  limit=_limit,
                                  offset=_offset)
        if len(order_by):
            query = query.order_by(*order_by)

        conn = self.db.executable
        if _streamed:
            conn = self.db.engine.connect()
            conn = conn.execution_options(stream_results=True)
            
        return ResultIter(conn.execute(query),
                          row_type=self.db.row_type,
                          step=_step)


class DatabaseExt(dataset.Database):
    """A DatabaseExt object represents a SQL database with  multiple tables of type TableExt."""

    """Extends the dataset.Database class and adds a
    flask_sqlalchemy.SQLAlchemy reference. Connects to a
    flask_sqlalchemy.
    """

    def __init__(self, url, flask_app=None, schema=None, reflect_metadata=True,
                 engine_kwargs=None, reflect_views=True,
                 ensure_schema=True, row_type=row_type):
        """Configure and connect to the database."""
        if url is None:
            url = os.environ.get('DATABASE_URL', 'sqlite://')
        if engine_kwargs is None:
            engine_kwargs = {}
        parsed_url = urlparse(url)
        if type(flask_app) is Flask:
            app = flask_app
        else:
            if flask_app is not None:
                app = Flask(flask_app)
            else:
                app = None
        if parsed_url.scheme.lower() in 'sqlite':
            # ref: https://github.com/pudo/dataset/issues/163
            if 'poolclass' not in engine_kwargs:
                engine_kwargs.config['poolclass'] = StaticPool
        engine_kwargs['SQLALCHEMY_DATABASE_URI'] = url
        if app:
            app.config['SQLALCHEMY_DATABASE_URI'] = url
            self.flask_db = SQLAlchemy(app, engine_options=engine_kwargs)
        else:
            self.flask_db = SQLAlchemy(engine_options=engine_kwargs)            
        # how do we work with session
        self.engine = self.flask_db.engine
        self.flask_db._engine_lock = self.lock = threading.RLock() # we are going to use a re-entrant lock because that's what dataset uses.
        self.local = threading.local()

        if len(parsed_url.query):
            query = parse_qs(parsed_url.query)
            if schema is None:
                schema_qs = query.get('schema', query.get('searchpath', []))
                if len(schema_qs):
                    schema = schema_qs.pop()

        self.types = dataset.types.Types()
        self.schema = schema
        self.url = url
        self.row_type = row_type
        self.ensure_schema = ensure_schema
        self._tables = {}

    # will only work for sqlite. 
    # diferent databases have different fts. 
    def create_fts_index_column(self, table_name, column, stemmer="unicode61"): #  porter 
        # maybe we create a mirror sqlite database called fts_db if the database we are opening is not of sqlite type.
        # the idea is we want to be able to locally attach fts with our datasets arrow files. 
        self.db.executeable.execute('CREATE VIRTUAL TABLE {table_name}_idx USING FTS5(idx:INTEGER, {column}:VARCHAR, tokenize="{stemmer}");')

    def create_table(self, table_name, primary_id=None, primary_type=None):
        """Create a new table.

        Either loads a table or creates it if it doesn't exist yet. You can
        define the name and type of the primary key field, if a new table is to
        be created. The default is to create an auto-incrementing integer,
        ``id``. You can also set the primary key to be a string or big integer.
        The caller will be responsible for the uniqueness of ``primary_id`` if
        it is defined as a text type.

        Returns a :py:class:`Table <dataset.Table>` instance.
        ::

            table = db.create_table('population')

            # custom id and type
            table2 = db.create_table('population2', 'age')
            table3 = db.create_table('population3',
                                     primary_id='city',
                                     primary_type=db.types.text)
            # custom length of String
            table4 = db.create_table('population4',
                                     primary_id='city',
                                     primary_type=db.types.string(25))
            # no primary key
            table5 = db.create_table('population5',
                                     primary_id=False)
        """
        assert not isinstance(primary_type, six.string_types), \
            'Text-based primary_type support is dropped, use db.types.'
        try:
            self.flask_db.create_all() # TODO, don't call this if we already called 
        except:
            pass
        table_name = dataset.util.normalize_table_name(table_name)
        with self.lock:
            if table_name not in self._tables:
                self._tables[table_name] = TableExt(self, table_name,
                                                 primary_id=primary_id,
                                                 primary_type=primary_type,
                                                 auto_create=True)
            return self._tables.get(table_name)

    def load_table(self, table_name):
        """Load a table.

        This will fail if the tables does not already exist in the database. If
        the table exists, its columns will be reflected and are available on
        the :py:class:`Table <dataset.Table>` object.

        Returns a :py:class:`Table <dataset.Table>` instance.
        ::

            table = db.load_table('population')
        """
        try:
            self.flask_db.create_all() # TODO, don't call this if we already called. how to sync the ._tables variable with the corresponding variable in 
        except:
            pass
        table_name = dataset.util.normalize_table_name(table_name)
        with self.lock:
            if table_name not in self._tables:
                self._tables[table_name] = TableExt(self, table_name)
            return self._tables.get(table_name)


class IndexGzipFileExt(igzip.IndexedGzipFile):
    """This class inheriets from `` ingdex_gzip.IndexedGzipFile``. This class allows in addition to the functionality 
    of IndexedGzipFile, access to a specific line based on the seek point of the line, using the __getitem__ method.

    Additionally, a (conginguous) list or slice can be used, which will be more efficient then doing line by line access. 
    
    The base IndexedGzipFile class allows for fast random access of a gzip
    file by using the ``zran`` library to build and maintain an index of seek
    points into the file.
    ``IndexedGzipFile`` is an ``io.BufferedReader`` which wraps an
    :class:`_IndexedGzipFile` instance. By accessing the ``_IndexedGzipFile``
    instance through an ``io.BufferedReader``, read performance is improved
    through buffering, and access to the I/O methods is made thread-safe.
    A :meth:`pread` method is also implemented, as it is not implemented by
    the ``io.BufferedReader``.
    """


    def __init__(self, *args, **kwargs):
        """Create an ``LineIndexGzipFile``. The file may be specified either
        with an open file handle (``fileobj``), or with a ``filename``. If the
        former, the file must have been opened in ``'rb'`` mode.
        .. note:: The ``auto_build`` behaviour only takes place on calls to
                  :meth:`seek`.
        :arg filename:         File name or open file handle.
        :arg fileobj:          Open file handle.
        :arg mode:             Opening mode. Must be either ``'r'`` or ``'rb``.
        :arg auto_build:       If ``True`` (the default), the index is
                               automatically built on calls to :meth:`seek`.
        :arg skip_crc_check:   Defaults to ``False``. If ``True``, CRC/size
                               validation of the uncompressed data is not
                               performed.
        :arg spacing:          Number of bytes between index seek points.
        :arg window_size:      Number of bytes of uncompressed data stored with
                               each seek point.
        :arg readbuf_size:     Size of buffer in bytes for storing compressed
                               data read in from the file.
        :arg readall_buf_size: Size of buffer in bytes used by :meth:`read`
                               when reading until EOF.
        :arg drop_handles:     Has no effect if an open ``fid`` is specified,
                               rather than a ``filename``.  If ``True`` (the
                               default), a handle to the file is opened and
                               closed on every access. Otherwise the file is
                               opened at ``__cinit__``, and kept open until
                               this ``_IndexedGzipFile`` is destroyed.
        :arg index_file:       Pre-generated index for this ``gz`` file -
                               if provided, passed through to
                               :meth:`import_index`.
        :arg buffer_size:      Optional, must be passed as a keyword argument.
                               Passed through to
                               ``io.BufferedReader.__init__``. If not provided,
                               a default value of 1048576 is used.
        :arg line2seekpoint:      Optional, must be passed as a keyword argument.
                               If not passed, this will automatically be created.                               
        """
        self.line2seekpoint        = kwargs.pop('line2seekpoint', None)
        super(LineIndexGzipFile, self).__init__(*args, **kwargs)
        pos = self.tell()
        self.seek(0, os.SEEK_END)
        self.file_size = file_size = self.tell() 
        self.seek(pos, 0)

        if self.line2seekpoint is None:
          def reader(fobj, rng, max_rng, ret):
            fobj.seek(rng)
            start = -1
            while rng < max_rng:
              fobj.readline()
              pos = fobj.tell()
              if start != -1 and pos < max_rng:
                ret.append(pos)
                start = pos
              rng = pos

          workers=[]
          line_nums = []
          for rng in range(0, file_size, 10000000):                    
            max_rng = min(rng + 10000000, file_size)
            line_nums.append([])
            worker = threading.Thread(target=reader, args=(copy.copy(self), rng, max_rng, line_nums[-1]))
            workers.append(worker)
            worker.start()
          for worker in workers:
            worker.join()
          self.line2seekpoint = itertools.chain(*line_nums)

    def __reduce__(self):
        """Used to pickle an ``LineIndexGzipFile``.
        Returns a tuple containing:
          - a reference to the ``unpickle`` function
          - a tuple containing a "state" object, which can be passed
            to ``unpickle``.
        """

        fobj = self.__igz_fobj

        if (not fobj.drop_handles) or (not fobj.own_file):
            raise pickle.PicklingError(
                'Cannot pickle IndexedGzipFile that has been created '
                'with an open file object, or that has been created '
                'with drop_handles=False')

        # export and serialise the index if
        # any index points have been created.
        # The index data is serialised as a
        # bytes object.
        if fobj.npoints == 0:
            index = None

        else:
            index = io.BytesIO()
            self.export_index(fileobj=index)
            index = index.getvalue()

        state = {
            'filename'         : fobj.filename,
            'auto_build'       : fobj.auto_build,
            'spacing'          : fobj.spacing,
            'window_size'      : fobj.window_size,
            'readbuf_size'     : fobj.readbuf_size,
            'readall_buf_size' : fobj.readall_buf_size,
            'buffer_size'      : self.__buffer_size,
            'line2seekpoint'   : fobj.line2seekpoint,
            'file_size'   : fobj.file_size,
            'tell'             : self.tell(),
            'index'            : index}

        return (unpickle, (state, ))

    
    def __getiter__(self):
        len_self = len(self)
        for start in range(0, len_self, 10000):
          end = min(len_self, rng+10000)
          start = self.line2seekpoint[start]
          if end == len_self:
            end = self.file_size
          else:
            end= self.line2seekpoint[end+1]-1
          ret = []
          with self.__file_lock:
            pos = self.tell()
            self.seek(0, start)
            ret= self.read(end-start).decode().split('\n')
            self.seek(pos, 0)
          for line in ret:
            yield line

    def __len__(self):
        return len(self.line2seekpoint)

    def __getitem__(self, keys):
        start, end = None, None
        if isinstance(keys, int):
          contiguous = False
        else:
          contiguous, start, end = is_contiguous(keys)
        if isinstance(keys, slice):
          contiguous = True
          start = 0 if keys.start is None else keys.start
          end = len(self) if keys.stop is None else keys.stop

        if contiguous:
          start = self.line2seekpoint[start]
          if end >= len(self.line2seekpoint):
            end = self.file_size
          else:
            end= self.line2seekpoint[end+1]-1
          with self.__file_lock:
            pos = self.tell()
            self.seek(0, start)
            ret= self.read(end-start).decode().split('\n')
            self.seek(pos, 0)
            return ret
        elif isinstance(keys, int):
          with self.__file_lock:
            pos = self.tell()
            self.seek(0, start)
            ret= self.readline().decode()
            self.seek(pos, 0)
            return ret
        else:
          return [self[idx] for idx in keys]

class FeaturesWithViews(Features):
    def copy(self):
        ret= FeaturesWithViews(super().copy())
        if hasattr(self, "features_map"):
            ret.features_map = copy.deepcopy(self.features_map)
        return ret

    def __repr__(self):
        ret =  "{"+"\n".join([f"'{a[0]}': {a[1]}" for a in self.items() if a[0] not in self.features_map])
        if self.features_map:
            ret = ret+"\n"+"\n".join(f"'{a[0]}': View({a[1]})" for a in  self.features_map.items())
        ret +="}"
        return ret


class Datastore(Dataset): #, dict
    """
    A class that wraps a Huggingface arrow based Dataset to provide some optimized reading and *writing* in various persistance backends. 
    Currently provides support for columns bound to dask memmap, txt file, and sqlalchemy, faiss, elastic search, minhash and apache beam.
    """
    @property 
    def features(self):
        ret = FeaturesWithViews(self._info.features)
        ret.features_map = {} if not hasattr(self, "features_map") else self.features_map
        return ret
        
    def __repr__(self):
        return f"Datastore({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}\n}})"
        
    @classmethod
    def from_dataset(cls, dataset, features_map=None, shared_dir=None):
        self = cls(
            arrow_table=dataset._data,
            indices_table=dataset._indices,
            info=dataset._info,
            split=dataset._split,
            fingerprint=dataset._fingerprint,
        )

        if  hasattr(dataset, "mmap_access_cnt"):
          self.mmap_access_cnt=dataset.mmap_access_cnt
        else:
          self.mmap_access_cnt=0
        if  hasattr(dataset, "features_map"):
          self.features_map=copy.deepcopy(dataset.features_map)
        if features_map is not None:
          self.features_map = copy.deepcopy(features_map)
        if not hasattr(self, "features_map"):
          self.features_map = {}
        if  hasattr(dataset, "shared_dir"):
          self.shared_dir=shared_dir
        if shared_dir is not None:
          self.shared_dir = shared_dir
        if not hasattr(self, "shared_dir"):
          self.shared_dir = {}
        return self

                             
    def _get_mmap(self, mmap_file_path,  dtype, shape):
      if shape[0] < len(self):
          shape[0] = len(self)
      # what happens when the datastore shrinks??
      if os.path.exists(mmap_file_path):
        ret= np.memmap(filename=mmap_file_path, mode="r+", dtype=np.dtype(dtype), shape=tuple(shape))
      else:
        ret = np.memmap(filename=mmap_file_path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
      if self.mmap_access_cnt % 100==0: #let's flush intermittently just in case the OS needs to synch.
        ret.flush()
      self.mmap_access_cnt+=1
      return ret

    # we use class variables because we don't want it serialized in an instance of this dataset
    igzip_fobj = {}
    def _get_igzip_fobj(self, file_path):
        if file_path in igzip_fobj:
            return igzip_fobj[file_path]
        igzip_fobj[file_path] = fobj = get_igzip_obj(file_path)
        return fobj

    # we use class variables because we don't want it serialized in this instance
    db_table = {}
    db_connection = {}
    def _get_db_table(self, table_name, connection_url):
        if (table_name, connection_url) in db_table:
            table =  db_table[(table_name, connection_url)]
        else:
            if connection_url in db_connection:
                db =  db_connection[connection_url]
            else:
                db_connection[connection_url] = db = DatabaseExt(connection_url)
            db_table[(table_name, connection_url)] = table = db[table_name]
        return table

    @staticmethod
    def _add_idx(batch, indices, idx,):
        batch[idx] = indices # will this be shuffled if we are in shuffled mode?
        return batch

    #mapping a columun to a memmap array accessed by row
    def set_mmap_feature_view(self, feature_view, shape, mmap_path=None, dtype='float32', dtype_str_len=1000, idx_column="id", batch_size=100000, num_proc=4, map_fn=None):
      dataset_path = os.path.dirname(self.cache_files[0]['filename'])
      if mmap_path is None:
               mmap_path = os.path.abspath(os.path.join(dataset_path, feature_view+".mmap"))
      shape = list(shape)
      shape[0] = len(self)
      if idx_column not in self.features:
        self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'idx': idx_column})
      if not isinstance(dtype, str):
          dtype =np.dtype(dtype).name
      self.features_map[feature_view] = {'type':"mmap", 'path': mmap_path,  'dtype': dtype, 'shape': shape}
      return self

    #mapping a column to an indexed gzip file accesed by line 
    def set_igzip_feature_view(self, feature_view, path,  idx_column="id", batch_size=100000, num_proc=4, map_fn=None):
      fobj = self._get_igzip_fobj(path)
      if idx_column not in self.features:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'idx': idx_column})
      if len(fobj) > len(self):
            self.add_item({idx_column: range(learn(self), len(fobj))})
      self.features_map[feature_view] = {'type':"igzip", 'path': path}
      return self

    # mapping columns to a sql database. creates a sqlalchmey/dataset dynamically with idx_column as the primary key. 
    def set_sql_feature_view(self, table_name, connection_url, columns=None, idx_column="id",  batch_size=100000, num_proc=4, map_fn=None):
        table = _get_db_table(table_name, connection_url)
        if table.columns:
            columns = table.columns
        elif not columns:
            raise RuntimeError(f"No column definition for table view {table_name}")
        if idx_column not in self.features:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'feature_view': idx_column})
        if len(table) > len(self):
            self.add_item({idx_column: range(len(self), len(table))})
        for col in columns:
            if col == idx_column:
                continue
            if col in self.features:
                raise RuntimeError(f"Column {col} already in the dataset")
            self.features_map[column] = {'type':'sql', 'connection_url': connection_url, 'table_name': table_name, 'column': column}
        return self
    
    def _getitem(
        self,
        key: Union[int, slice, str],
        format_type=None,
        format_columns=None,
        output_all_columns=False,
        format_kwargs=None,
    ) -> Union[Dict, List]:

        orig_format_type = format_type
        orig_key = key
        column = None
        if isinstance(key, str):
          column = key
        contiguous = None
        if isinstance(key, str) or (isinstance(key, slice) and key.step is None):
            contiguous=True
        if column is not None and column in self.features_map:
          if format_type in ("torch", "tensorflow", "numpy"):
            orig_format_type = format_type
            format_type = "numpy"
          elif format_type in (None, "custom"):
            orig_format_type = format_type
            format_type = None
          orig_column = column
          key = "id"

        missing=[]
        if format_columns:
             for c in copy.copy(format_columns):
                 if c in self.features_map:
                     missing.append(c)
                     format_columns.remove(c)
                     if "id" not in format_columns:
                         format_columns.append("id")

        if (hasattr(self, "features_map") and not self.features_map) and len(self.features) == 1 and "id" in self.features:
            # this dataset is empty so we don't want to return just ids that might not relate to any data
            if format_type in (None, "custom"):
                return {}
            elif format_type == "torch":
                import torch
                return torch.tensor([])
            elif format_type == "numpy":
                return np.array([])
            elif format_type == "pandas":
                return pd.DataFrame()
            elif format_type == "tensorflow":
                import tensorflow
                return tensorflow.ragged.constant([])
            return None

        if hasattr(self, "features_map") and (len(self.features) == 1 and "id" in self.features) and contiguous and (isinstance(key, str)):
            # we are only getting data from a column that is a view
            return self._format_views(orig_key, column=column, format_type=format_type, orig_format_type=orig_format_type,
                                     output_all_columns=output_all_columns, format_kwargs=format_kwargs, contiguous=contiguous)

        #let's get the data that is in the arrow file
        outputs = super()._getitem(
              key,
              format_type=format_type,
              format_columns=format_columns,
              output_all_columns=output_all_columns,
              format_kwargs=format_kwargs)

        # restore format_columns incase it's referenced somewhere else
        if format_columns and "id" in format_columns:
            format_columns.remove("id")

        if format_columns is not None:
            format_columns.extend(missing)

        # now combine views and retrieved arrow data 
        return self._format_views(outputs, column=column, format_type=format_type, orig_format_type=orig_format_type,
                                 output_all_columns=output_all_columns, format_kwargs=format_kwargs, contiguous=contiguous)
        
    def _format_views(self,  
        outputs_or_keys,       
        column=None,
        format_type=None,
        orig_format_type=None,
        format_columns=None,
        output_all_columns=False,
        format_kwargs=None,
        contiguous=None):

        def getitems(self, outputs, column, keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items):
            if column is None:
                items in self.features_map.items()
            else:
                items = (column, self.features_map[column])
            sql_results = {}
            for feature, val in items:
              if column is not None or (format_columns in (None, []) or feature in format_columns or output_all_columns):
                if val['type'] == 'mmap':
                    if mmap_by_items:
                        if contiguous:
                            outputs[feature] = [ self._get_mmap(val['path'], val['dtype'], val['shape']) for i in range(start, end)]
                        else:
                            outputs[feature] = [ self._get_mmap(val['path'], val['dtype'], val['shape']) for i in keys]
                    else:
                        if contiguous:
                            outputs[feature] = self._get_mmap(val['path'], val['dtype'], val['shape'])[start:end]
                        else:
                            outputs[feature] = self._get_mmap(val['path'], val['dtype'], val['shape'])[keys]                            
                elif val['type'] == 'igzip':
                    if contiguous:
                        outputs[feature] = self._get_igzip_fobj(val['path'])[start:end]
                    else:
                        outputs[feature] = self._get_igzip_fobj(val['path'])[keys]
                elif val['type'] == 'sql':
                    sql_results[(val['table_name'], val['connection_url'])] = sql_results.get((val['table_name'], val['connection_url']),[])+[feature]
            for table_connection, features in sql_results:
                table_name, connection_url = table_connection
                table= self._get_db_table(table_name, connection_url)
                if contiguous:
                    for row in table.find((table.id, 'between', (start, end)), _columns=features):
                        for feature in features:
                            outputs[feature] = output.get(feature,[]) + row[feature]
                else:
                    for row in table.find((table.id, 'in', keys), _columns=features):
                        for feature in features:
                            outputs[feature] = output.get(feature,[]) + row[feature]
            return outputs
        # todo, change the id key from "id" to something custom

        format_kwargs = format_kwargs if format_kwargs is not None else {}
        format_columns = format_columns if format_columns is not None else []
        if format_type in ("custom", None) or isinstance(output_or_keys, dict): 
            transform = format_kwargs.get('transform')
            if isinstance(output_or_keys, str):
                keys = slice(0, len(self))
                outputs = {}
                contiguous=True
            elif isinstance(outputs_or_keys, slice):
                keys = outputs_or_keys
                outputs = {}
                contiguous=True
            else:
                keys = outputs_or_keys["id"]
                outputs = outputs_or_keys
            if not contiguous:
                  if isinstance(keys, int):
                        contiguous = False
                  else:
                        contiguous, start, end = is_contiguous(keys)
            else:
                  if isinstance(keys, slice):
                    start = 0 if keys.start is None else keys.start
                    end = len(self) if keys.stop is None else keys.stop
                  else:
                    start = keys[0]
                    end = keys[-1]+1
            outputs = getitems(self, outputs, column, keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=False)
            if transform is not None:
              outputs = transform(outputs)
            if "id" in outputs: del outputs["id"] # what if the user specifically asked for "id"?
            return outputs
        elif format_type == "pandas" or isinstance(outputs_or_keys, pd.DataFrame):
            # do we do transforms for this case??
            if isinstance(outputs_or_keys, str):
                start = 0 
                end = len(self) 
                keys = range(start, stop)
                outputs = None
                contiguous=True
            elif isinstance(outputs_or_keys, slice):
                start = 0 if outputs_or_keys.start is None else outputs_or_keys.start
                end = len(self) if outputs_or_keys.stop is None else outputs_or_keys.stop
                keys = range(outputs_or_keys.start, outputs_or_keys.stop)
                outputs = None
                contiguous=True
            elif isinstance(outputs_or_keys, dict) or isinstance(outputs_or_keys,  pd.DataFrame):
                outputs = outputs_or_keys
                outputs = pd.DataFrame(outputs)
                keys = outputs_or_keys["id"]
                contiguous, start, end = is_contiguous(keys)
            else:
                raise RuntimeError("got unknown outputs or keys type")
            if outputs is None:
                outputs = pd.DataFrame()
            outputs = getitems(self, outputs, column, keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=True)
            if column is not None:
                outputs = outputs[column]
            else:
                if "id" in outputs: outputs.drop("id", axis=1) # what if the user specifically asked for "id"?
            return outputs
        elif column is not None and column in self.features_map:
            # do we do transforms for this case??
            val = self.features_map[column]
            if isinstance(outputs, str):
                  contiguous = True
                  start = 0 
                  end = len(self)
                  keys = range(0, len(self))
            elif isinstance(outputs, slice):
                  contiguous = True
                  start = 0 if outputs.start is None else outputs.start
                  end = len(self) if outputs.stop is None else outputs.stop
                  keys = range(0, len(self))
            elif not contiguous:
                if isinstance(outputs, int):
                  contiguous = False
                  keys = outputs
                else:
                  contiguous, start, end = is_contiguous(outputs)
            else:
                start= outputs[0]
                end= outputs[-1]+1
            if contiguous:
              x= self.get_view_data(column, val)[start:end]
            else:
              x= self.get_view_data(column, val)[keys]
            if orig_format_type == "torch":
              import torch
              return torch.tensor(x, **format_kwargs)
            elif format_type == "tensorflow":
              import tensorflow
              return tensorflow.ragged.constant(x, **format_kwargs)
            else:
              return x
        return outputs

    def to_csv(
        self,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        **to_csv_kwargs,
    ) -> int:
      pass

    def to_dict(self, batch_size: Optional[int] = None, batched: bool = False) -> Union[dict, Iterator[dict]]:
        if (not hasattr(self, "features_map") or not self.features_map) and len(self.features) == 1 and "id" in self.features:
            return {}
        #TODO - put back direct mmap access method here?
        ret = super().to_dict(batch_size=batch_size, batched=batched)
        if isinstance(ret, Iterator):
            for r in ret:
                yield self._format_views(r, contiguous=True)
            return 
        return self._format_views(ret, contiguous=True)

    def to_pandas(
        self, batch_size: Optional[int] = None, batched: bool = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        if (not hasattr(self, "features_map") or not self.features_map) and len(self.features) == 1 and "id" in self.features:
            return pd.DataFrame()
        #TODO - put back direct mmap access method here?
        ret = super().to_pandas(batch_size=batch_size, batched=batched)
        if isinstance(ret, Iterator):
            for r in ret:
                yield self._format_views(r, contiguous=True)
        return self._format_views(ret, contiguous=True)
        
    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "cache_file_name"])
    def _map_single_old(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        new_fingerprint: Optional[str] = None,
        rank: Optional[int] = None,
        offset: int = 0,
        update_data=True,
    ) -> "Dataset":
      ret = super()._map_single(function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            new_fingerprint=new_fingerprint,
            rank=rank,
            offset=offset,
            update_data=update_data,)
      features_map= copy.deepcopy(self.features_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in features_map:
              del features_map[column]
      return Datastore.from_dataset(ret, features_map=features_map)

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":

      ret = super().map(function=function, with_indices=with_indices, input_columns=input_columns,
                     batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                     remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                     load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                     writer_batch_size=writer_batch_size, features=features,
                     disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                     num_proc=num_proc, suffix_template=suffix_template,
                     new_fingerprint=new_fingerprint)
      features_map= copy.deepcopy(self.features_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in features_map:
              del features_map[column]
      return Datastore.from_dataset(ret, features_map=features_map)


    def class_encode_column(self, column: str) -> "Datastore":
        if column in self.features_map:
            raise NotImplementedError()
        ret = super().class_encode_column(column)
        return Datastore.from_dataset(ret, features_map=self.features_map)
    
    @fingerprint_transform(inplace=False)
    def flatten(self, new_fingerprint, max_depth=16) -> "Datastore":
        ret = super().flatten(new_fingerprint, max_depth)
        return Datastore.from_dataset(ret, features_map=self.features_map)

    def cast(
        self,
        features: Features,
        batch_size: Optional[int] = 10_000,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 10_000,
        num_proc: Optional[int] = None,
    ) -> "Datastore":
        for feature in self.features_map:
            if feature not in features:
                continue
            if  self.features[feature] != features[feature]:
                raise NotImplementedError()
        ret = super().cast(
          features =features,
          batch_size = batch_size ,
          keep_in_memory = keep_in_memory,
          load_from_cache_file = load_from_cache_file,
          cache_file_name = cache_file_name,
          writer_batch_size = writer_batch_size,
          num_proc = num_proc)
        return Datastore.from_dataset(ret, features_map=self.features_map)

    @fingerprint_transform(inplace=False)
    def remove_columns(self, column_names: Union[str, List[str]], new_fingerprint) -> "Datastore":
        ret = super().remove_columns(column_names=column_names, new_fingerprint=new_fingerprint)
        features_map= copy.deepcopy(self.features_map)
        for column in [column_names] if instance(column_names, str) else column_names:
            if column in features_map:
                del features_map[column]
        return Datastore.from_dataset(ret, features_map=features_map)

    @fingerprint_transform(inplace=False)
    def rename_column(self, original_column_name: str, new_column_name: str, new_fingerprint) -> "Datastore":
        ret = super().rename_column(original_column_name=original_column_name, new_column_name=new_column_name, new_fingerprint=new_fingerprint)
        features_map= copy.deepcopy(self.features_map)
        if original_column_name in features_map:
            val = features_map[original_column_name]
            del features_map[original_column_name]
            features_map[new_column_name] = val
        return Datastore.from_dataset(ret, features_map=features_map)


    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "cache_file_name"])
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().filter(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batch_size=batch_size,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint)
        features_map= copy.deepcopy(self.features_map)
        for column in remove_columns if remove_columns is not None else []:
            if column in features_map:
                del features_map[column]
        return Datastore.from_dataset(ret, features_map=features_map)

    #@replayable_table_alteration
    @fingerprint_transform(inplace=False, ignore_kwargs=["cache_file_name"])
    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = True,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().flatten_indices(
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            new_fingerprint=new_fingerprint,
            )
        return Datastore.from_dataset(ret, features_map=self.features_map)

    
    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def select_new(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            ) 
        return Datastore.from_dataset(ret, features_map=self.features_map)

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"])
    def sort(
        self,
        column: str,
        reverse: bool = False,
        kind: str = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        if column in self.features_map:
            raise NotImplementedError()
        ret = super().sort(
            column=column,
            reverse=reverse,
            kind=kind,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )
        return Datastore.from_dataset(ret, features_map=self.features_map)



    @transmit_format
    @fingerprint_transform(
        inplace=False, randomized_function=True, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"]
    )
    def shuffle(
        self,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().shuffle(
            seed=seed,
            generator=generator,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            )
        return Datastore.from_dataset(ret, features_map=self.features_map)
  
    @transmit_format
    @fingerprint_transform(
        inplace=False,
        randomized_function=True,
        fingerprint_names=["train_new_fingerprint", "test_new_fingerprint"],
        ignore_kwargs=["load_from_cache_file", "train_indices_cache_file_name", "test_indices_cache_file_name"],
    )
    def train_test_split(
        self,
        test_size: Union[float, int, None] = None,
        train_size: Union[float, int, None] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        train_indices_cache_file_name: Optional[str] = None,
        test_indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        train_new_fingerprint: Optional[str] = None,
        test_new_fingerprint: Optional[str] = None,
    ) -> "DatastoreDict":
        ret = super.train_test_split(
            test_size=test_size,
            train_size=train_size,
            shuffle=shuffle,
            seed=seed,
            generator=generator,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            train_indices_cache_file_name=train_indices_cache_file_name,
            test_indices_cache_file_name=test_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            train_new_fingerprint=train_new_fingerprint,
            test_new_fingerprint=test_new_fingerprint,
        )
        for key in list(ret.keys()):
            ret[key] = Datastore.from_dataset(ret, features_map=self.features_map)
        return ret

    # shard doesn't seem to work properly because of pickling problems? Maybe it's because it's being run in Colab with autoload??
    def shard_new(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = False,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "Datastore":
        ret = super().shard(num_shards=num_shards,
          index=index,
          contiguous=contiguous,
          keep_in_memory=keep_in_memory,
          indices_cache_file_name=indices_cache_file_name,
          writer_batch_size=writer_batch_size)
        return ret # Datastore.from_dataset(ret, features_map=self.features_map)


    # TODO: Fiix load_from_idsk and save_to_disk to work with current version of datasets

    @staticmethod
    def load_from_disk(dataset_path: str, fs=None, shared_dir=None) -> "Datastore":
      # TODO, move from shared drive to cached drive
        ret = Dataset.load_from_disk(dataset_path=dataset_path, fs=fs)
        dataset_path = os.path.dirname(ret._data_files[0]["filename"])
        with open(
            Path(dataset_path, "state.json").as_posix(), "r", encoding="utf-8"
        ) as state_file:
            state = json.load(state_file)
        ret.features_map =  state.get("features_map")
        for key, values in list(ret.features_map.items()):
            mmap_path = os.path.abspath(os.path.join(dataset_path, values[0]))
            ret.features_map[key][0] =  mmap_path
        return Datastore.from_dataset(ret)
        #todo, do periodic sync with the shared drive, and lazy loading of shareds from shared drive

    def save_to_disk(self, dataset_path: str, move_files=True):
        """
        Save the datastore along with all mmaps and uris in a directory

        Args:
            dataset_path (``str``): path of the dataset directory where the dataset will be saved to
        """
        assert (
            not self.list_indexes()
        ), "please remove all the indexes using `dataset.drop_index` before saving a dataset"
        orig_self = self
        if not move_files:
            self = pickle.loads(pickle.dumps(self))
        os.makedirs(dataset_path, exist_ok=True)
        orig_dataset_path = os.path.dirname(self._data_files[0]["filename"])
        # Write indices if needed
        if self._indices is not None:
            if not self._indices_data_files:
                cache_file_name = os.path.join(dataset_path, "indices.arrow")
                writer = ArrowWriter(path=cache_file_name)
                writer.write_table(self._indices)
                writer.finalize()
                self._indices_data_files = [{"filename": cache_file_name}]
        # Write dataset if needed
        if not self._data_files or any(len(h["transforms"]) > 0 for h in self._inplace_history):
            cache_file_name = os.path.join(dataset_path, "dataset.arrow")
            writer = ArrowWriter(path=cache_file_name)
            writer.write_table(self._data)
            writer.finalize()
            self._data_files = [{"filename": cache_file_name}]
            self._inplace_history = [{"transforms": []}]
        # Copy all files into the dataset directory
        for data_file in self._data_files + self._indices_data_files :
            # Copy file to destination directory
            src = data_file["filename"]
            filename = Path(src).name
            dest = os.path.join(dataset_path, filename)
            if src != dest:
                shutil.move(src, dest)
            # Change path to relative path from inside the destination directory
            data_file["filename"] = filename
        for key, value in list(self.features_map.items()):
            # Copy file to destination directory
            src = value[0]
            filename = Path(src).name
            dest = os.path.join(dataset_path, filename)
            # if the src is not under the 
            if src != dest and os.path.exists(src):
                if filename.startswith(orig_dataset_path):
                  shutil.move(src, dest)
                else:
                  shutil.copy(src, dest)
            # Change path to relative path from inside the destination directory
            self.features_map[key] = [filename]  + value[1:]
        if not move_files:
          return orig_self
        # Get state
        state = self.__getstate__()
        dataset_info = json.loads(state.pop("_info"))
        assert state.get("_data") is None, "arrow table needs to be memory mapped"
        assert state.get("_indices") is None, "arrow table needs to be memory mapped"
        assert all(
            len(h["transforms"]) == 0 for h in state.get("_inplace_history", [])
        ), "in-place history needs to be empty"
        # Serialize state
        with open(os.path.join(dataset_path, "state.json"), "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
        with open(os.path.join(dataset_path, "dataset_info.json"), "w", encoding="utf-8") as dataset_info_file:
            json.dump(dataset_info, dataset_info_file, indent=2, sort_keys=True)
#        logger.info("Dataset saved in {}".format(dataset_path))
        for key, values in list(self.features_map.items()):
            mmap_path = os.path.abspath(os.path.join(dataset_path, values[0]))
            self.features_map[key][0] =  mmap_path
        return self

    


