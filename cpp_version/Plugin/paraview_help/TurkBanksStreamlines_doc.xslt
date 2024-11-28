<xml>
    <proxy>
        <group>filters</group>
        <name>TurkBanksFilter</name>
        <label>Turk Banks Streamlines</label>
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
            <name>MaxIterations</name>
            <label>MaxIterations</label>
            <documentation>
                <brief/>
                <long>
          The number of iterations before the live source stop.
        </long>
            </documentation>
            <defaults>2000</defaults>
            <domains/>
        </property>
    </proxy>
</xml>
