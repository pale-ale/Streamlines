<xml>
<proxy>
    <group>filters</group>
    <name>TurkBanksFilter</name>
    <label>Turk Banks Streamline Filter</label>
    <documentation>
        <brief>Live source dummy.</brief>
        <long/>
    </documentation>
    <property>
        <name>Vector Field Data</name>
        <label>Vector Field Data</label>
        <documentation>
            <brief/>
            <long>
          Set the vector field we create streamlines in.
        </long>
        </documentation>
        <defaults/>
        <domains>
            <domain>
                <text>Accepts input of following types:</text>
                <list>
                    <item>vtkImageData</item>
                </list>
            </domain>
        </domains>
    </property>
    <property>
        <name>Iterations</name>
        <label>Iterations</label>
        <defaults>100</defaults>
        <domains/>
    </property>
    <property>
        <name>JoinDistanceFactor</name>
        <label>Join Distance Factor</label>
        <defaults>1.0</defaults>
        <domains/>
    </property>
    <property>
        <name>Separation</name>
        <label>Separation</label>
        <defaults>0.04</defaults>
        <domains/>
    </property>
    <property>
        <name>BirthThreshold</name>
        <label>Birth Threshold</label>
        <defaults>0.0</defaults>
        <domains/>
    </property>
    <property>
        <name>IntegrationStepSize</name>
        <label>Integration Step Size</label>
        <defaults>0.005</defaults>
        <domains/>
    </property>
    <property>
        <name>CoherenceWeight</name>
        <label>Coherence Weight</label>
        <defaults>0.5</defaults>
        <domains/>
    </property>
    <property>
        <name>ResolutionFactor</name>
        <label>Resolution Factor</label>
        <defaults>1.0</defaults>
        <domains/>
    </property>
    <property>
        <name>TemporalFilterRadius</name>
        <label>Temporal Filter Radius</label>
        <defaults>0.7</defaults>
        <domains/>
    </property>
    <property>
        <name>TimestepValues</name>
        <label>TimestepValues</label>
        <documentation>
            <brief/>
            <long>
             Available timestep values.
         </long>
        </documentation>
        <defaults/>
        <domains/>
    </property>
    <property>
        <name>Enable Shattering</name>
        <label>Enable Shattering</label>
        <defaults>1</defaults>
        <domains>
            <domain>
                <text>Accepts boolean values (0 or 1).</text>
            </domain>
        </domains>
    </property>
    <property>
        <name>SingleFrameMode</name>
        <label>SingleFrameMode</label>
        <defaults>1</defaults>
        <domains>
            <domain>
                <text>Accepts boolean values (0 or 1).</text>
            </domain>
        </domains>
    </property>
</proxy>

</xml>
